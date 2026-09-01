#!/usr/bin/env python3
"""Immutable local run records for RL training.

Phase-1 goals:
- do not change algorithm mathematics;
- create one immutable run directory per invocation;
- capture resolved config, Git/system metadata, timing and throughput;
- mirror the existing models/<technique>/train_log.csv into a run-local CSV;
- print periodic progress/ETA without requiring trainer changes;
- preserve best/final/recovery model artifacts for later evaluation;
- never depend on W&B.

The class intentionally monitors the training CSV already produced by train.py,
so existing trainer function signatures do not need to change in Phase 1.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import traceback
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import yaml
except Exception as exc:  # pragma: no cover - project already depends on PyYAML
    raise RuntimeError("run_context.py requires PyYAML") from exc


DEFAULT_PROGRESS_INTERVAL_S = 60.0
POLL_INTERVAL_S = 2.0


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe_slug(value: str, max_len: int = 48) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value).strip())
    value = re.sub(r"-+", "-", value).strip("-._")
    return (value or "run")[:max_len]


def _format_duration(seconds: Optional[float]) -> str:
    if seconds is None or seconds < 0 or seconds == float("inf"):
        return "?"
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _run_command(args: List[str], cwd: Path, timeout: float = 3.0) -> Optional[str]:
    try:
        p = subprocess.run(
            args,
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
            check=False,
        )
        out = p.stdout.strip()
        return out if out else None
    except Exception:
        return None


def _package_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except Exception:
        return None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=False, default=str)
        f.write("\n")
    os.replace(tmp, path)


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            value = json.load(f)
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def _read_training_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def _read_last_training_row(path: Path, tail_bytes: int = 8192) -> Optional[Dict[str, str]]:
    """Read only the header and tail of a growing CSV.

    The progress monitor calls this every few seconds, so it must not rescan an
    entire 100k/200k-row training log on each poll. A full-file read is still
    performed once at finalization, where the cost is appropriate.
    """
    try:
        if not path.exists() or path.stat().st_size == 0:
            return None

        with path.open("rb") as f:
            header_raw = f.readline()
            if not header_raw:
                return None
            header = next(csv.reader([header_raw.decode("utf-8").rstrip("\r\n")]))
            if not header:
                return None

            f.seek(0, os.SEEK_END)
            size = f.tell()
            if size <= len(header_raw):
                return None
            read_size = min(int(tail_bytes), size)
            f.seek(size - read_size)
            tail = f.read(read_size).decode("utf-8", errors="ignore")

        lines = [line for line in tail.splitlines() if line.strip()]
        if not lines:
            return None

        # If the tail starts in the middle of a row, ignore that first fragment.
        # Work backwards so a concurrently-written partial final row falls back
        # to the most recent complete row instead of breaking monitoring.
        for line in reversed(lines):
            try:
                values = next(csv.reader([line]))
            except Exception:
                continue
            if len(values) != len(header):
                continue
            row = dict(zip(header, values))
            if row.get("episode") not in (None, ""):
                return row
        return None
    except Exception:
        return None


def _to_int(row: Dict[str, str], key: str) -> Optional[int]:
    try:
        return int(float(row[key]))
    except Exception:
        return None


def _to_float(row: Dict[str, str], key: str) -> Optional[float]:
    try:
        return float(row[key])
    except Exception:
        return None


class RunContext:
    """Capture one train.py invocation as an immutable local experiment record."""

    def __init__(
        self,
        *,
        technique: str,
        tech_config: Dict[str, Any],
        full_config: Dict[str, Any],
        model_dir: Path,
        force: bool,
        run_name: Optional[str] = None,
        repo_root: Optional[Path] = None,
        runs_root: Optional[Path] = None,
    ) -> None:
        self.technique = str(technique)
        self.tech_config = deepcopy(tech_config)
        self.full_config = deepcopy(full_config)
        self.model_dir = Path(model_dir).resolve()
        self.force = bool(force)
        self.run_name = str(run_name).strip() if run_name else None
        self.repo_root = Path(repo_root or Path(__file__).resolve().parent).resolve()
        self.runs_root = Path(runs_root or (self.repo_root / "runs")).resolve()

        stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        host = _safe_slug(socket.gethostname(), 24)
        suffix = uuid.uuid4().hex[:6]
        label = f"_{_safe_slug(self.run_name, 32)}" if self.run_name else ""
        self.run_id = f"{stamp}_{host}_{_safe_slug(self.technique, 24)}{label}_{suffix}"
        self.run_dir = self.runs_root / self.technique / self.run_id
        self.artifacts_dir = self.run_dir / "artifacts"

        self.started_monotonic: Optional[float] = None
        self.started_epoch_s: Optional[float] = None
        self.started_at: Optional[str] = None
        self.completed_at: Optional[str] = None
        self._stop_event = threading.Event()
        self._monitor_thread: Optional[threading.Thread] = None
        self._last_print_time = 0.0
        self._last_print_progress: Optional[int] = None

        self.log_path = self.model_dir / "train_log.csv"
        self.metadata_path = self.model_dir / "training_metadata.json"
        self.initial_metadata = {} if self.force else _read_json(self.metadata_path)
        self.initial_rows = [] if self.force else _read_training_rows(self.log_path)
        self.initial_row_count = 0 if self.force else len(self.initial_rows)

        self.eval_metrics_path = self.model_dir / "eval_metrics.csv"
        self.evaluations_path = self.model_dir / "evaluations"
        if self.force:
            self.initial_eval_row_count = 0
            self.initial_evaluation_files = set()
        else:
            try:
                with self.eval_metrics_path.open(newline="", encoding="utf-8") as f:
                    self.initial_eval_row_count = sum(1 for _ in csv.DictReader(f))
            except Exception:
                self.initial_eval_row_count = 0
            self.initial_evaluation_files = (
                {p.name for p in self.evaluations_path.glob("*.json")}
                if self.evaluations_path.exists() else set()
            )

        self.target_unit, self.target_total = self._target()
        self.start_progress = self._initial_progress()
        self.progress_interval_s = float(
            self.full_config.get("run_progress_interval_s", DEFAULT_PROGRESS_INTERVAL_S)
        )
        if self.progress_interval_s <= 0:
            self.progress_interval_s = DEFAULT_PROGRESS_INTERVAL_S

        self.manifest: Dict[str, Any] = {}
        self._pre_run_git: Optional[Dict[str, Any]] = None
        self._pre_run_system: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------
    def __enter__(self) -> "RunContext":
        # Capture repository/system state BEFORE creating runs/<...>, otherwise
        # the run record itself would make an otherwise-clean Git tree dirty.
        self._pre_run_git = self._git_info()
        self._pre_run_system = self._system_info()

        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        self.started_monotonic = time.monotonic()
        self.started_epoch_s = time.time()
        self.started_at = _now_iso()

        self._write_resolved_config()
        self.manifest = self._base_manifest()
        _atomic_json(self.run_dir / "manifest.json", self.manifest)

        print(f"  [run] id: {self.run_id}")
        print(f"  [run] record: {self.run_dir}")

        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            name=f"run-monitor-{self.technique}",
            daemon=True,
        )
        self._monitor_thread.start()
        return self

    def __exit__(self, exc_type, exc_value, exc_tb) -> bool:
        self._stop_event.set()
        if self._monitor_thread is not None:
            self._monitor_thread.join(timeout=5.0)

        status = "completed"
        failure = None
        if exc_type is not None:
            # A user-requested Ctrl-C is not an algorithm failure. Preserve the
            # partial run as an interrupted experiment so it can be inspected or
            # resumed later, while still allowing KeyboardInterrupt to propagate
            # to the caller for the conventional exit status.
            status = "interrupted" if issubclass(exc_type, KeyboardInterrupt) else "failed"
            failure = {
                "type": getattr(exc_type, "__name__", str(exc_type)),
                "message": str(exc_value),
                "traceback": "".join(traceback.format_exception(exc_type, exc_value, exc_tb)),
            }

        try:
            self._finalize(status=status, failure=failure)
        except Exception as finalize_exc:
            # Experiment recording must never hide the actual training result.
            print(f"  [run] WARNING: could not finalize run record: {finalize_exc}")

        return False

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------
    def _target(self) -> Tuple[str, Optional[int]]:
        if "total_timesteps" in self.tech_config:
            try:
                return "timesteps", int(self.tech_config["total_timesteps"])
            except Exception:
                return "timesteps", None
        if "num_episodes" in self.tech_config:
            try:
                return "episodes", int(self.tech_config["num_episodes"])
            except Exception:
                return "episodes", None
        return "unknown", None

    def _initial_progress(self) -> int:
        if self.force:
            return 0
        key = "steps_completed" if self.target_unit == "timesteps" else "episodes_completed"
        try:
            return int(self.initial_metadata.get(key, 0) or 0)
        except Exception:
            pass

        if self.initial_rows:
            row = self.initial_rows[-1]
            parsed = _to_int(row, "timestep" if self.target_unit == "timesteps" else "episode")
            if parsed is not None:
                return parsed
        return 0

    def _git_info(self) -> Dict[str, Any]:
        head = _run_command(["git", "rev-parse", "HEAD"], self.repo_root)
        branch = _run_command(["git", "rev-parse", "--abbrev-ref", "HEAD"], self.repo_root)
        status_all = _run_command(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            self.repo_root,
        ) or ""

        # Generated experiment/viewing outputs do not affect training source.
        ignored_prefixes = (
            "runs/", "wandb/", "submissions/", ".dvc/cache/", ".dvc/tmp/",
            "__pycache__/",
        )
        ignored_suffixes = (".log", ".patch")
        relevant_lines = []
        for line in status_all.splitlines():
            path = line[3:].strip() if len(line) >= 4 else line.strip()
            # Handle rename records: old -> new; the destination is enough for
            # generated-path filtering, while tracked renames remain relevant.
            filter_path = path.split(" -> ")[-1]
            if any(filter_path.startswith(prefix) for prefix in ignored_prefixes):
                continue
            if line.startswith("?? ") and filter_path.endswith(ignored_suffixes):
                continue
            relevant_lines.append(line)

        return {
            "commit": head,
            "branch": branch,
            "dirty": bool(relevant_lines),
            "status_porcelain": "\n".join(relevant_lines),
            "status_porcelain_all": status_all,
        }

    def _system_info(self) -> Dict[str, Any]:
        gpu_names = _run_command(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            self.repo_root,
            timeout=2.0,
        )
        return {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor() or None,
            "cpu_count": os.cpu_count(),
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
            "requested_device": self.tech_config.get("device"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "nvidia_gpu_names": gpu_names.splitlines() if gpu_names else [],
            "packages": {
                "numpy": _package_version("numpy"),
                "torch": _package_version("torch"),
                "stable_baselines3": _package_version("stable-baselines3"),
                "gymnasium": _package_version("gymnasium"),
                "wandb": _package_version("wandb"),
            },
        }

    def _base_manifest(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "run_id": self.run_id,
            "run_name": self.run_name,
            "technique": self.technique,
            "portal_name": self.tech_config.get("portal_name"),
            "category": self.tech_config.get("category"),
            "status": "running",
            "force_fresh": self.force,
            "command": [sys.executable] + sys.argv,
            "command_text": " ".join([sys.executable] + sys.argv),
            "started_at": self.started_at,
            "started_at_utc": _utc_iso(),
            "completed_at": None,
            "wall_time_s": None,
            "target": {
                "unit": self.target_unit,
                "total": self.target_total,
                "start": self.start_progress,
            },
            "git": self._pre_run_git or self._git_info(),
            "system": self._pre_run_system or self._system_info(),
            "paths": {
                "repo_root": str(self.repo_root),
                "model_dir": str(self.model_dir),
                "run_dir": str(self.run_dir),
            },
            "artifacts": {},
            "summary_file": "summary.json",
            "resolved_config_file": "resolved_config.yaml",
            "metrics_file": "train_metrics.csv",
            "eval_metrics_file": "eval_metrics.csv",
            "failure": None,
        }

    def _write_resolved_config(self) -> None:
        global_config = {
            k: deepcopy(v)
            for k, v in self.full_config.items()
            if k != "techniques"
        }
        resolved = {
            "technique": self.technique,
            "technique_config": deepcopy(self.tech_config),
            "global_config": global_config,
        }
        with (self.run_dir / "resolved_config.yaml").open("w", encoding="utf-8") as f:
            yaml.safe_dump(resolved, f, sort_keys=False, default_flow_style=False)

    # ------------------------------------------------------------------
    # Progress monitor / ETA
    # ------------------------------------------------------------------
    def _current_progress(self) -> Optional[int]:
        if self.force and self.started_epoch_s is not None:
            try:
                if self.log_path.stat().st_mtime + 0.5 < self.started_epoch_s:
                    return None
            except OSError:
                return None

        row = _read_last_training_row(self.log_path)
        if not row:
            return None

        key = "timestep" if self.target_unit == "timesteps" else "episode"
        return _to_int(row, key)

    def _progress_snapshot(self) -> Optional[Dict[str, Any]]:
        if self.started_monotonic is None:
            return None

        # Read the latest complete trainer row once.  Besides progress, the
        # existing train_log.csv contains trainer wall_time_s.  Prefer that for
        # throughput/ETA so imports, W&B startup, RunContext setup, etc. do not
        # make the first ETA wildly pessimistic.  Fall back to total experiment
        # elapsed time for trainers/logs that do not expose wall_time_s.
        if self.force and self.started_epoch_s is not None:
            try:
                if self.log_path.stat().st_mtime + 0.5 < self.started_epoch_s:
                    return None
            except OSError:
                return None

        row = _read_last_training_row(self.log_path)
        if not row:
            return None

        key = "timestep" if self.target_unit == "timesteps" else "episode"
        current = _to_int(row, key)
        if current is None:
            return None

        elapsed = max(time.monotonic() - self.started_monotonic, 1e-9)
        trainer_elapsed = _to_float(row, "wall_time_s")
        if trainer_elapsed is not None and trainer_elapsed > 0:
            rate_elapsed = trainer_elapsed
            rate_basis = "trainer"
        else:
            rate_elapsed = elapsed
            rate_basis = "total"

        gained = max(current - self.start_progress, 0)
        rate = gained / rate_elapsed if gained > 0 and rate_elapsed > 0 else 0.0
        remaining = None
        eta = None
        pct = None
        if self.target_total is not None:
            remaining = max(self.target_total - current, 0)
            raw_pct = 100.0 * current / self.target_total if self.target_total > 0 else 100.0
            # SB3 rollout-based algorithms may legitimately finish beyond the
            # requested timestep budget (for example 10,000 -> 12,288). Keep
            # the real completed count in the run record, but never display
            # progress above 100%.
            pct = min(raw_pct, 100.0)
            eta = remaining / rate if rate > 0 else None

        return {
            "current": current,
            "target": self.target_total,
            "start": self.start_progress,
            "elapsed_s": elapsed,
            "trainer_elapsed_s": trainer_elapsed,
            "rate_basis": rate_basis,
            "rate_per_s": rate,
            "remaining": remaining,
            "eta_s": eta,
            "percent": pct,
        }

    def _print_progress(self, snap: Dict[str, Any], force_print: bool = False) -> None:
        now = time.monotonic()
        current = int(snap["current"])
        if not force_print:
            if now - self._last_print_time < self.progress_interval_s:
                return
            if self._last_print_progress == current:
                return

        unit = "steps" if self.target_unit == "timesteps" else "ep"
        target = snap.get("target")
        pct = snap.get("percent")
        if target is not None and pct is not None:
            display_current = min(current, int(target))
            progress_text = f"{display_current:,}/{int(target):,} {unit} ({pct:5.1f}%)"
        else:
            progress_text = f"{current:,} {unit}"

        rate = float(snap.get("rate_per_s") or 0.0)
        rate_text = f"{rate:,.1f} {unit}/s" if rate > 0 else "rate=?"
        print(
            f"  [run] {progress_text} | elapsed={_format_duration(snap.get('elapsed_s'))} "
            f"| {rate_text} | ETA={_format_duration(snap.get('eta_s'))}"
        )
        self._last_print_time = now
        self._last_print_progress = current

    def _monitor_loop(self) -> None:
        while not self._stop_event.wait(POLL_INTERVAL_S):
            try:
                snap = self._progress_snapshot()
                if snap:
                    self._print_progress(snap)
            except Exception:
                # Monitoring is diagnostic only and must never affect training.
                continue

    # ------------------------------------------------------------------
    # Finalization
    # ------------------------------------------------------------------
    def _extract_run_metrics(self) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        all_rows = _read_training_rows(self.log_path)
        if not all_rows:
            return [], []

        if self.force:
            run_rows = all_rows
        else:
            # Episode is the most reliable cross-family boundary because every
            # existing train_log.csv row has episode + timestep fields.
            start_episode = 0
            try:
                start_episode = int(self.initial_metadata.get("episodes_completed", 0) or 0)
            except Exception:
                start_episode = 0
            if start_episode > 0:
                run_rows = [
                    row for row in all_rows
                    if (_to_int(row, "episode") or 0) > start_episode
                ]
            else:
                run_rows = all_rows[self.initial_row_count :]

        return all_rows, run_rows

    def _write_metrics_csv(self, run_rows: List[Dict[str, str]]) -> None:
        out = self.run_dir / "train_metrics.csv"
        if not run_rows:
            out.write_text("episode,timestep,episode_cost,wall_time_s,timestamp\n", encoding="utf-8")
            return

        fieldnames: List[str] = []
        for row in run_rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
        with out.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(run_rows)

    def _rolling_summary(
        self,
        all_rows: List[Dict[str, str]],
        run_rows: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        costs: List[float] = []
        parsed: List[Tuple[Dict[str, str], float]] = []
        for row in all_rows:
            cost = _to_float(row, "episode_cost")
            if cost is not None:
                costs.append(cost)
                parsed.append((row, cost))

        run_episode_set = {
            _to_int(row, "episode") for row in run_rows if _to_int(row, "episode") is not None
        }

        best_roll = None
        best_row = None
        best_index = None
        window: List[float] = []
        for idx, (row, cost) in enumerate(parsed):
            window.append(cost)
            if len(window) > 100:
                window.pop(0)
            if len(window) < 100:
                continue
            ep = _to_int(row, "episode")
            if run_rows and ep not in run_episode_set:
                continue
            avg = sum(window) / len(window)
            if best_roll is None or avg < best_roll:
                best_roll = avg
                best_row = row
                best_index = idx

        final_roll = None
        if costs:
            tail = costs[-100:]
            final_roll = sum(tail) / len(tail)

        trainer_meta = _read_json(self.metadata_path)
        result = {
            "episodes_logged_this_run": len(run_rows),
            "rows_logged_total": len(all_rows),
            "best_observed_rolling100_cost": best_roll,
            "final_observed_rolling100_cost": final_roll,
            "trainer_best_rolling_cost": trainer_meta.get("best_rolling_cost"),
            "best_episode": _to_int(best_row, "episode") if best_row else None,
            "best_timestep": _to_int(best_row, "timestep") if best_row else None,
            "time_to_best_s": _to_float(best_row, "wall_time_s") if best_row else None,
            "best_row_index": best_index,
            "trainer_metadata": trainer_meta,
        }
        return result

    def _copy_artifacts(self) -> Dict[str, Any]:
        copied: Dict[str, Any] = {}
        candidates: List[Path] = []
        for pattern in (
            "best_model.*",
            "best_eval_model.*",
            "final_model.*",
            "hyperparams_used.yaml",
            "training_metadata.json",
            "evaluation_metadata.json",
            "discretizer_config.json",
        ):
            candidates.extend(sorted(self.model_dir.glob(pattern)))

        ckpt_dir = self.model_dir / "checkpoints"
        if ckpt_dir.exists():
            for pattern in ("checkpoint.*", "replay_buffer.pkl"):
                candidates.extend(sorted(ckpt_dir.glob(pattern)))

        seen = set()
        for src in candidates:
            if not src.is_file() or src.resolve() in seen:
                continue
            seen.add(src.resolve())
            if src.parent == ckpt_dir:
                rel = Path("checkpoints") / src.name
            else:
                rel = Path(src.name)
            dst = self.artifacts_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(src, dst)
                copied[str(rel)] = {
                    "size_bytes": dst.stat().st_size,
                    "sha256": _sha256(dst),
                }
            except Exception as exc:
                copied[str(rel)] = {"copy_error": str(exc)}
        return copied

    def _copy_evaluation_records(self) -> Dict[str, Any]:
        """Copy only this invocation's periodic-evaluation records."""
        copied = {
            "metrics": False, "rows": 0, "detail_files": 0,
            "evaluation_wall_time_s": 0.0,
        }
        src_csv = self.model_dir / "eval_metrics.csv"
        if src_csv.exists() and src_csv.is_file():
            try:
                with src_csv.open(newline="", encoding="utf-8") as f:
                    rows = list(csv.DictReader(f))
                run_rows = rows[self.initial_eval_row_count:]
                if run_rows:
                    dst = self.run_dir / "eval_metrics.csv"
                    with dst.open("w", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=list(run_rows[0].keys()))
                        writer.writeheader()
                        writer.writerows(run_rows)
                    copied["metrics"] = True
                    copied["rows"] = len(run_rows)
                    total_eval_wall = 0.0
                    for row in run_rows:
                        try:
                            total_eval_wall += float(row.get("eval_wall_time_s", 0) or 0)
                        except (TypeError, ValueError):
                            pass
                    copied["evaluation_wall_time_s"] = total_eval_wall
            except Exception as exc:
                copied["metrics_error"] = str(exc)

        src_dir = self.model_dir / "evaluations"
        if src_dir.exists() and src_dir.is_dir():
            dst_dir = self.run_dir / "evaluations"
            for src in sorted(src_dir.glob("*.json")):
                if not src.is_file() or src.name in self.initial_evaluation_files:
                    continue
                try:
                    dst_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst_dir / src.name)
                    copied["detail_files"] += 1
                except Exception:
                    pass
        return copied

    def _finalize(self, *, status: str, failure: Optional[Dict[str, Any]]) -> None:
        assert self.started_monotonic is not None
        wall = time.monotonic() - self.started_monotonic
        self.completed_at = _now_iso()

        # One final ETA/progress line when progress exists.
        snap = self._progress_snapshot()
        if snap:
            self._print_progress(snap, force_print=True)

        all_rows, run_rows = self._extract_run_metrics()
        self._write_metrics_csv(run_rows)
        perf = self._rolling_summary(all_rows, run_rows)
        artifacts = self._copy_artifacts()
        evaluation_records = self._copy_evaluation_records()
        evaluation_meta = _read_json(self.model_dir / "evaluation_metadata.json")

        final_meta = _read_json(self.metadata_path)
        final_steps = final_meta.get("steps_completed")
        final_episodes = final_meta.get("episodes_completed")
        if final_steps is None and all_rows:
            values = [_to_int(row, "timestep") for row in all_rows]
            values = [v for v in values if v is not None]
            final_steps = max(values) if values else None
        if final_episodes is None and all_rows:
            values = [_to_int(row, "episode") for row in all_rows]
            values = [v for v in values if v is not None]
            final_episodes = max(values) if values else None

        start_steps = 0 if self.force else int(self.initial_metadata.get("steps_completed", 0) or 0)
        start_episodes = 0 if self.force else int(self.initial_metadata.get("episodes_completed", 0) or 0)
        delta_steps = max(int(final_steps or 0) - start_steps, 0)
        delta_episodes = max(int(final_episodes or 0) - start_episodes, 0)

        if status == "completed" and delta_steps == 0 and delta_episodes == 0:
            status = "no_op"

        trainer_wall = perf.get("trainer_metadata", {}).get("wall_time_s")
        try:
            trainer_wall = float(trainer_wall) if trainer_wall is not None else None
        except (TypeError, ValueError):
            trainer_wall = None
        framework_overhead = (
            max(wall - trainer_wall, 0.0)
            if trainer_wall is not None and trainer_wall >= 0
            else None
        )
        evaluation_wall = float(
            evaluation_records.get("evaluation_wall_time_s", 0.0) or 0.0)
        learning_wall = (
            max(trainer_wall - evaluation_wall, 0.0)
            if trainer_wall is not None and trainer_wall >= 0
            else None
        )

        summary = {
            "run_id": self.run_id,
            "technique": self.technique,
            "status": status,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            # End-to-end experiment occupancy, including setup/finalization.
            "wall_time_s": wall,
            "total_wall_time_s": wall,
            # Trainer-reported learning-loop time, when the trainer provides it.
            "trainer_wall_time_s": trainer_wall,
            "framework_overhead_s": framework_overhead,
            "evaluation_wall_time_s": evaluation_wall,
            "learning_wall_time_s": learning_wall,
            "start_steps": start_steps,
            "completed_steps": final_steps,
            "steps_this_run": delta_steps,
            "steps_per_second": delta_steps / wall if wall > 0 and delta_steps > 0 else None,
            "steps_per_second_training": (
                delta_steps / trainer_wall
                if trainer_wall is not None and trainer_wall > 0 and delta_steps > 0
                else None
            ),
            "steps_per_second_learning_only": (
                delta_steps / learning_wall
                if learning_wall is not None and learning_wall > 0 and delta_steps > 0
                else None
            ),
            "start_episodes": start_episodes,
            "completed_episodes": final_episodes,
            "episodes_this_run": delta_episodes,
            "episodes_per_second": delta_episodes / wall if wall > 0 and delta_episodes > 0 else None,
            "episodes_per_second_training": (
                delta_episodes / trainer_wall
                if trainer_wall is not None and trainer_wall > 0 and delta_episodes > 0
                else None
            ),
            "episodes_per_second_learning_only": (
                delta_episodes / learning_wall
                if learning_wall is not None and learning_wall > 0 and delta_episodes > 0
                else None
            ),
            "best_eval_mean_cost": evaluation_meta.get("best_mean_cost"),
            "evaluations_completed": evaluation_meta.get("evaluations_completed", 0),
            "evaluation_selection_episodes": evaluation_meta.get(
                "episodes_per_evaluation"),
            **perf,
        }
        _atomic_json(self.run_dir / "summary.json", summary)

        self.manifest.update(
            {
                "status": status,
                "completed_at": self.completed_at,
                "wall_time_s": wall,
                "failure": failure,
                "artifacts": artifacts,
                "evaluation_records": evaluation_records,
                "result": {
                    "steps_completed": final_steps,
                    "episodes_completed": final_episodes,
                    "steps_this_run": delta_steps,
                    "episodes_this_run": delta_episodes,
                    "total_wall_time_s": summary["total_wall_time_s"],
                    "trainer_wall_time_s": summary["trainer_wall_time_s"],
                    "framework_overhead_s": summary["framework_overhead_s"],
                    "evaluation_wall_time_s": summary["evaluation_wall_time_s"],
                    "learning_wall_time_s": summary["learning_wall_time_s"],
                    "steps_per_second": summary["steps_per_second"],
                    "steps_per_second_training": summary["steps_per_second_training"],
                    "steps_per_second_learning_only": summary[
                        "steps_per_second_learning_only"],
                    "episodes_per_second": summary["episodes_per_second"],
                    "episodes_per_second_training": summary["episodes_per_second_training"],
                    "episodes_per_second_learning_only": summary[
                        "episodes_per_second_learning_only"],
                    "trainer_best_rolling_cost": perf.get("trainer_best_rolling_cost"),
                    "best_observed_rolling100_cost": perf.get("best_observed_rolling100_cost"),
                    "final_observed_rolling100_cost": perf.get("final_observed_rolling100_cost"),
                    "best_episode": perf.get("best_episode"),
                    "best_timestep": perf.get("best_timestep"),
                    "time_to_best_s": perf.get("time_to_best_s"),
                    "best_eval_mean_cost": evaluation_meta.get("best_mean_cost"),
                    "evaluations_completed": evaluation_meta.get(
                        "evaluations_completed", 0),
                },
            }
        )
        _atomic_json(self.run_dir / "manifest.json", self.manifest)

        print(
            f"  [run] {status}: wall={_format_duration(wall)} "
            f"steps={delta_steps:,} episodes={delta_episodes:,}"
        )
        best = perf.get("trainer_best_rolling_cost")
        if best is not None:
            try:
                print(f"  [run] trainer best rolling cost: {float(best):,.3f}")
            except Exception:
                pass
        print(f"  [run] saved record: {self.run_dir}")
