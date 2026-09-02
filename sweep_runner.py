#!/usr/bin/env python3
"""Controlled, isolated hyperparameter sweep runner.

Each trial trains into its own output root, so parallel trials never overwrite
``models/<technique>`` or DVC-tracked champion artifacts.  Local CSV/JSON files
remain authoritative; W&B is optional and disabled by default for sweeps.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import queue
import re
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent


def _slug(text: str) -> str:
    out = "".join(c if c.isalnum() or c in "-_" else "-" for c in text)
    return out.strip("-") or "sweep"


def _load(path: Path) -> Dict[str, Any]:
    with path.open() as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("sweep spec must be a YAML mapping")
    return data


def _encode(value: Any) -> str:
    # JSON is valid YAML and is unambiguous for CLI round-tripping.
    return json.dumps(value, separators=(",", ":"))


def _expand_trials(spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    base = dict(spec.get("base_overrides", {}) or {})
    grid = dict(spec.get("grid", {}) or {})
    repeats = max(int(spec.get("repeats", 1)), 1)
    for key, values in grid.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"grid value for {key!r} must be a non-empty list")

    keys = list(grid)
    combos = itertools.product(*(grid[k] for k in keys)) if keys else [()]
    trials = []
    idx = 0
    for combo in combos:
        combo_overrides = dict(base)
        combo_overrides.update(dict(zip(keys, combo)))
        for repeat in range(repeats):
            idx += 1
            trials.append({
                "index": idx,
                "repeat": repeat,
                "overrides": combo_overrides,
            })
    return trials


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Atomically replace a JSON control file read by dashboards/sync jobs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2))
    os.replace(temporary, path)


def _trial_result(trial_dir: Path, technique: str) -> Dict[str, Any]:
    model_dir = trial_dir / "models" / technique
    training = _read_json(model_dir / "training_metadata.json")
    evaluation = _read_json(model_dir / "evaluation_metadata.json")
    plateau = evaluation.get("plateau", {}) if isinstance(evaluation, dict) else {}
    return {
        "steps_completed": training.get("steps_completed"),
        "episodes_completed": training.get("episodes_completed"),
        "best_training_rolling_cost": training.get("best_rolling_cost"),
        "best_eval_cost": evaluation.get("best_mean_cost"),
        "plateau_detected": plateau.get("detected"),
        "plateau_first_progress": plateau.get("first_detected_progress"),
    }


def _write_summary(root: Path, rows: List[Dict[str, Any]]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "sweep_summary.json").write_text(json.dumps(rows, indent=2))
    fields = [
        "trial_id", "returncode", "wall_time_s", "device",
        "steps_completed", "episodes_completed",
        "best_training_rolling_cost", "best_eval_cost",
        "plateau_detected", "plateau_first_progress", "overrides",
    ]
    with (root / "sweep_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: r["trial_id"]):
            payload = {k: row.get(k) for k in fields}
            payload["overrides"] = json.dumps(payload.get("overrides", {}), sort_keys=True)
            writer.writerow(payload)


def _summary_rows(root: Path) -> List[Dict[str, Any]]:
    """Rebuild a complete sweep summary without dropping untouched trials."""
    rows: List[Dict[str, Any]] = []
    for trial_dir in sorted(root.glob("trial_[0-9][0-9][0-9]")):
        manifest = _read_json(trial_dir / "trial.json")
        if not manifest:
            continue
        technique = str(manifest.get("technique", ""))
        if not technique:
            continue
        rows.append({
            "trial_id": manifest.get("trial_id", trial_dir.name),
            "returncode": manifest.get("returncode"),
            "wall_time_s": manifest.get("wall_time_s"),
            "device": manifest.get("device"),
            "overrides": manifest.get("overrides", {}),
            **_trial_result(trial_dir, technique),
        })
    return rows


def _resume_command(
    manifest: Dict[str, Any], trial_dir: Path, target_kind: str, target: int,
) -> List[str]:
    original = manifest.get("command")
    if not isinstance(original, list) or not original:
        raise ValueError(f"existing trial has no usable command: {trial_dir / 'trial.json'}")
    cmd = [str(value) for value in original if str(value) != "--force"]

    expected_flag = "--timesteps" if target_kind == "steps" else "--episodes"
    other_flag = "--episodes" if expected_flag == "--timesteps" else "--timesteps"
    if other_flag in cmd:
        raise ValueError(
            f"resume target does not match original command ({other_flag} is present)"
        )
    if expected_flag in cmd:
        index = cmd.index(expected_flag)
        if index + 1 >= len(cmd):
            raise ValueError(f"missing value after {expected_flag} in existing command")
        cmd[index + 1] = str(target)
    else:
        cmd += [expected_flag, str(target)]

    if "--output-root" not in cmd:
        raise ValueError("existing command has no --output-root")
    output_index = cmd.index("--output-root")
    if output_index + 1 >= len(cmd):
        raise ValueError("missing value after --output-root in existing command")
    actual_output = Path(cmd[output_index + 1]).expanduser().resolve()
    expected_output = (trial_dir / "models").resolve()
    if actual_output != expected_output:
        raise ValueError(
            f"refusing resume: command output root is {actual_output}, "
            f"expected {expected_output}"
        )
    return cmd


def _parse_resume_log(path: Path | None) -> Dict[str, Any]:
    details: Dict[str, Any] = {}
    if path is None or not path.is_file():
        return details
    text = path.read_text(errors="replace")
    run_match = re.search(r"\[run\] id: (\d{8}_\d{6})_", text)
    if run_match:
        details["started_at"] = datetime.strptime(
            run_match.group(1), "%Y%m%d_%H%M%S"
        ).isoformat()
    wall_match = re.search(r"\[run\] completed: wall=(\d+)m(\d+)s", text)
    if wall_match:
        details["wall_time_s"] = (
            int(wall_match.group(1)) * 60 + int(wall_match.group(2))
        )
    if details.get("started_at") and details.get("wall_time_s") is not None:
        completed = datetime.fromisoformat(details["started_at"]) + timedelta(
            seconds=float(details["wall_time_s"])
        )
        details["completed_at"] = completed.isoformat()
    return details


def _append_reconciled_log(stdout_path: Path, source_path: Path) -> None:
    source = str(source_path.resolve())
    marker = f"===== RECONCILED CONTINUATION source={source} ====="
    existing = stdout_path.read_text(errors="replace") if stdout_path.exists() else ""
    if marker in existing:
        return
    source_bytes = source_path.read_bytes()
    with stdout_path.open("a") as output:
        if existing and not existing.endswith("\n"):
            output.write("\n")
        output.write(f"\n{marker}\n")
        output.write(source_bytes.decode(errors="replace"))
        if source_bytes and not source_bytes.endswith(b"\n"):
            output.write("\n")


def _resume_existing(args: argparse.Namespace) -> int:
    root = args.resume_sweep.expanduser().resolve()
    trial_id = str(args.resume_trial or "")
    if not re.fullmatch(r"trial_[0-9]{3}", trial_id):
        raise ValueError("--resume-trial must look like trial_001")
    trial_dir = root / trial_id
    manifest_path = trial_dir / "trial.json"
    manifest = _read_json(manifest_path)
    if not manifest:
        raise FileNotFoundError(f"existing trial manifest not found: {manifest_path}")

    targets = [
        ("steps", args.target_timesteps),
        ("episodes", args.target_episodes),
    ]
    targets = [(kind, value) for kind, value in targets if value is not None]
    if len(targets) != 1:
        raise ValueError(
            "resume requires exactly one of --target-timesteps / --target-episodes"
        )
    target_kind, target_value = targets[0]
    target = int(target_value)
    if target <= 0:
        raise ValueError("resume target must be positive")

    technique = str(manifest["technique"])
    model_dir = trial_dir / "models" / technique
    training = _read_json(model_dir / "training_metadata.json")
    progress_key = "steps_completed" if target_kind == "steps" else "episodes_completed"
    completed = training.get(progress_key)
    if completed is None:
        raise ValueError(f"training metadata has no {progress_key}: {model_dir}")
    completed = int(completed)
    if not args.reconcile_only and target <= completed:
        raise ValueError(
            f"target {target:,} must be greater than completed progress {completed:,}"
        )

    checkpoint_dir = model_dir / "checkpoints"
    checkpoints = list(checkpoint_dir.glob("checkpoint.*"))
    if not checkpoints:
        raise FileNotFoundError(f"resume checkpoint missing under {checkpoint_dir}")

    cmd = _resume_command(manifest, trial_dir, target_kind, target)
    print(f"Resume sweep: {root.name}")
    print(f"Trial: {trial_id}")
    print(f"Technique: {technique}")
    print(f"Progress: {completed:,} -> {target:,} {target_kind}")
    print("Force: disabled")
    print("Command: " + " ".join(cmd))
    if args.dry_run:
        print("DRY RUN: no files changed and no process started")
        return 0

    if "initial_run" not in manifest:
        manifest["initial_run"] = {
            key: manifest.get(key) for key in (
                "command", "started_at", "completed_at", "returncode",
                "wall_time_s", "steps_completed", "episodes_completed",
                "best_training_rolling_cost", "best_eval_cost",
                "plateau_detected", "plateau_first_progress",
            )
        }
    continuations = list(manifest.get("continuations", []))

    source_log = args.resume_log.expanduser().resolve() if args.resume_log else None
    source_text = str(source_log) if source_log else None
    if args.reconcile_only and source_text:
        for existing in continuations:
            if existing.get("source_log") == source_text:
                print(f"Already reconciled: {source_text}")
                _write_summary(root, _summary_rows(root))
                return 0

    now = datetime.now().isoformat()
    entry: Dict[str, Any] = {
        "sequence": len(continuations) + 1,
        "mode": "reconciled" if args.reconcile_only else "sweep_runner_resume",
        "command": cmd,
        "source_log": source_text,
        f"from_{target_kind}": manifest.get(progress_key, completed),
        f"target_{target_kind}": target,
        "started_at": now,
    }

    if args.reconcile_only:
        parsed = _parse_resume_log(source_log)
        entry.update(parsed)
        reconcile_rc = args.reconcile_returncode
        if reconcile_rc is None:
            reconcile_rc = 0 if completed >= target else 130
        reconcile_rc = int(reconcile_rc)
        reconcile_state = (
            "COMPLETED" if reconcile_rc == 0 and completed >= target
            else "INTERRUPTED" if reconcile_rc in (130, -signal.SIGINT)
            else "FAILED"
        )
        entry["returncode"] = reconcile_rc
        entry["status"] = reconcile_state
        entry["reconciled_at"] = now
        result = _trial_result(trial_dir, technique)
        entry.update(result)
        continuations.append(entry)
        manifest.update({
            "command": cmd,
            "returncode": reconcile_rc,
            "wall_time_s": entry.get("wall_time_s"),
            "started_at": entry.get("started_at", now),
            "completed_at": entry.get("completed_at", now),
            "continuations": continuations,
            "continuation_count": len(continuations),
            "state": reconcile_state,
            **result,
        })
        if source_log is not None:
            if not source_log.is_file():
                raise FileNotFoundError(f"resume log not found: {source_log}")
            _append_reconciled_log(trial_dir / "stdout.log", source_log)
        _write_json(manifest_path, manifest)
        _write_summary(root, _summary_rows(root))
        print(f"Reconciled {trial_id}; summary refreshed: {root / 'sweep_summary.csv'}")
        return 0

    entry["status"] = "RUNNING"
    manifest.update({
        "command": cmd,
        "started_at": now,
        "returncode": None,
        "completed_at": None,
        "state": "RUNNING",
        "active_continuation": entry,
    })
    _write_json(manifest_path, manifest)
    _write_summary(root, _summary_rows(root))

    env = os.environ.copy()
    resolved_spec = _load(root / "resolved_sweep.yaml")
    env["WANDB_MODE"] = str(resolved_spec.get("wandb_mode", "disabled"))
    env["PYTHONUNBUFFERED"] = "1"
    device = manifest.get("device")
    if device is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(device)

    stdout_path = trial_dir / "stdout.log"
    started = time.monotonic()
    interrupted = False
    with stdout_path.open("a") as log:
        log.write(
            f"\n===== SWEEP RUNNER CONTINUATION {now}: "
            f"{completed:,} -> {target:,} {target_kind} =====\n"
        )
        log.flush()
        proc = subprocess.Popen(
            cmd, cwd=PROJECT_ROOT, env=env,
            stdout=log, stderr=subprocess.STDOUT,
            text=True, start_new_session=True,
        )
        try:
            rc = int(proc.wait())
        except KeyboardInterrupt:
            interrupted = True
            os.killpg(proc.pid, signal.SIGINT)
            rc = int(proc.wait())

    wall = time.monotonic() - started
    result = _trial_result(trial_dir, technique)
    completed_at = datetime.now().isoformat()
    entry.update({
        "status": "INTERRUPTED" if interrupted else "COMPLETED",
        "returncode": rc,
        "wall_time_s": wall,
        "completed_at": completed_at,
        **result,
    })
    continuations.append(entry)
    manifest.update({
        "returncode": rc,
        "wall_time_s": wall,
        "completed_at": completed_at,
        "state": entry["status"] if interrupted else ("COMPLETED" if rc == 0 else "FAILED"),
        "continuations": continuations,
        "continuation_count": len(continuations),
        **result,
    })
    manifest.pop("active_continuation", None)
    _write_json(manifest_path, manifest)
    _write_summary(root, _summary_rows(root))
    print(f"[{trial_id}] resume done rc={rc} wall={wall:.1f}s")
    print(f"Summary: {root / 'sweep_summary.csv'}")
    return rc


def main() -> int:
    parser = argparse.ArgumentParser(description="Run isolated RL hyperparameter sweeps")
    parser.add_argument("spec", type=Path, nargs="?", help="YAML sweep specification")
    parser.add_argument("--max-parallel", type=int, default=None)
    parser.add_argument("--devices", default=None,
                        help="Comma-separated CUDA device IDs, e.g. 0,1")
    parser.add_argument("--shard-index", type=int, default=0,
                        help="Zero-based machine shard index")
    parser.add_argument("--shard-count", type=int, default=1,
                        help="Number of machine shards")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume-sweep", type=Path,
                        help="Reuse an existing timestamped sweep directory")
    parser.add_argument("--resume-trial",
                        help="Existing trial to continue, e.g. trial_002")
    parser.add_argument("--target-timesteps", type=int,
                        help="New total timestep target N2 for a resumed trial")
    parser.add_argument("--target-episodes", type=int,
                        help="New total episode target N2 for a resumed trial")
    parser.add_argument("--reconcile-only", action="store_true",
                        help="Record an already-run manual continuation without training")
    parser.add_argument("--resume-log", type=Path,
                        help="Manual continuation log to merge during reconciliation")
    parser.add_argument("--reconcile-returncode", type=int,
                        help="Override inferred return code for --reconcile-only")
    args = parser.parse_args()

    if args.resume_sweep is not None:
        return _resume_existing(args)
    if any((args.resume_trial, args.target_timesteps, args.target_episodes,
            args.reconcile_only, args.resume_log, args.reconcile_returncode)):
        parser.error("resume options require --resume-sweep")
    if args.spec is None:
        parser.error("spec is required for a fresh sweep")

    spec_path = args.spec.expanduser().resolve()
    spec = _load(spec_path)
    technique = str(spec["technique"])
    name = _slug(str(spec.get("name", spec_path.stem)))
    budget = dict(spec.get("budget", {}) or {})
    if set(budget) - {"episodes", "timesteps"}:
        raise ValueError("budget supports only 'episodes' or 'timesteps'")
    if len(budget) > 1:
        raise ValueError("choose only one of budget.episodes / budget.timesteps")

    shard_count = max(int(args.shard_count), 1)
    shard_index = int(args.shard_index)
    if not (0 <= shard_index < shard_count):
        raise ValueError("shard-index must satisfy 0 <= index < shard-count")

    all_trials = _expand_trials(spec)
    trials = [t for t in all_trials if (t["index"] - 1) % shard_count == shard_index]

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root_cfg = spec.get("output_root")
    if root_cfg:
        root = Path(root_cfg).expanduser()
        if not root.is_absolute():
            root = PROJECT_ROOT / root
        root = root.resolve() / f"{name}_{stamp}"
    else:
        root = PROJECT_ROOT / "sweeps" / f"{name}_{stamp}"
    root.mkdir(parents=True, exist_ok=True)
    (root / "resolved_sweep.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))

    devices_text = args.devices if args.devices is not None else spec.get("devices")
    if isinstance(devices_text, list):
        devices = [str(x) for x in devices_text]
    elif devices_text:
        devices = [x.strip() for x in str(devices_text).split(",") if x.strip()]
    else:
        devices = []

    max_parallel = args.max_parallel or int(spec.get("max_parallel", 1))
    max_parallel = max(max_parallel, 1)
    if devices:
        max_parallel = min(max_parallel, len(devices))

    print(f"Sweep: {name}")
    print(f"Technique: {technique}")
    print(f"Trials on this shard: {len(trials)} / {len(all_trials)}")
    print(f"Parallelism: {max_parallel}")
    print(f"Output: {root}")

    device_queue: queue.Queue[str] | None = None
    if devices:
        device_queue = queue.Queue()
        for dev in devices:
            device_queue.put(dev)

    wandb_mode = str(spec.get("wandb_mode", "disabled"))
    python_exe = str(spec.get("python", sys.executable))
    force = bool(spec.get("force", True))

    def run_trial(trial: Dict[str, Any]) -> Dict[str, Any]:
        trial_id = f"trial_{trial['index']:03d}"
        trial_dir = root / trial_id
        trial_dir.mkdir(parents=True, exist_ok=True)
        run_name = f"{name}-{trial_id}"
        cmd = [
            python_exe,
            str(PROJECT_ROOT / "train.py"),
            technique,
            "--run-name", run_name,
            "--output-root", str(trial_dir / "models"),
        ]
        if force:
            cmd.append("--force")
        if "episodes" in budget:
            cmd += ["--episodes", str(int(budget["episodes"]))]
        if "timesteps" in budget:
            cmd += ["--timesteps", str(int(budget["timesteps"]))]
        for key, value in trial["overrides"].items():
            cmd += ["--set", f"{key}={_encode(value)}"]

        device = None
        if device_queue is not None:
            device = device_queue.get()
        env = os.environ.copy()
        env["WANDB_MODE"] = wandb_mode
        if device is not None:
            env["CUDA_VISIBLE_DEVICES"] = device

        manifest = {
            "trial_id": trial_id,
            "technique": technique,
            "repeat": trial["repeat"],
            "overrides": trial["overrides"],
            "command": cmd,
            "device": device,
            "started_at": datetime.now().isoformat(),
        }
        (trial_dir / "trial.json").write_text(json.dumps(manifest, indent=2))
        print(f"[{trial_id}] start device={device or '-'} overrides={trial['overrides']}")
        started = time.monotonic()
        rc = 0
        try:
            if args.dry_run:
                (trial_dir / "stdout.log").write_text("DRY RUN\n" + " ".join(cmd) + "\n")
            else:
                with (trial_dir / "stdout.log").open("w") as log:
                    proc = subprocess.run(
                        cmd, cwd=PROJECT_ROOT, env=env,
                        stdout=log, stderr=subprocess.STDOUT,
                        text=True, check=False,
                    )
                rc = int(proc.returncode)
        finally:
            if device_queue is not None and device is not None:
                device_queue.put(device)
        wall = time.monotonic() - started
        row = {
            "trial_id": trial_id,
            "returncode": rc,
            "wall_time_s": wall,
            "device": device,
            "overrides": trial["overrides"],
            **(_trial_result(trial_dir, technique) if not args.dry_run else {}),
        }
        manifest.update(row)
        manifest["completed_at"] = datetime.now().isoformat()
        (trial_dir / "trial.json").write_text(json.dumps(manifest, indent=2))
        print(f"[{trial_id}] done rc={rc} wall={wall:.1f}s")
        return row

    rows: List[Dict[str, Any]] = []
    if max_parallel == 1:
        for trial in trials:
            rows.append(run_trial(trial))
            _write_summary(root, rows)
    else:
        with ThreadPoolExecutor(max_workers=max_parallel) as pool:
            futures = {pool.submit(run_trial, t): t for t in trials}
            for future in as_completed(futures):
                rows.append(future.result())
                _write_summary(root, rows)

    _write_summary(root, rows)
    failures = [r for r in rows if int(r.get("returncode", 1)) != 0]
    print(f"Completed {len(rows)} trial(s); failures={len(failures)}")
    print(f"Summary: {root / 'sweep_summary.csv'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
