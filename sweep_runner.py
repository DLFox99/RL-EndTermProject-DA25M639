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
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Run isolated RL hyperparameter sweeps")
    parser.add_argument("spec", type=Path, help="YAML sweep specification")
    parser.add_argument("--max-parallel", type=int, default=None)
    parser.add_argument("--devices", default=None,
                        help="Comma-separated CUDA device IDs, e.g. 0,1")
    parser.add_argument("--shard-index", type=int, default=0,
                        help="Zero-based machine shard index")
    parser.add_argument("--shard-count", type=int, default=1,
                        help="Number of machine shards")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

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
