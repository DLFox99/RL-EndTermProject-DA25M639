#!/usr/bin/env python3
"""Periodic deterministic evaluation and best-eval checkpoint selection."""

from __future__ import annotations

import csv
import json
import math
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional

from evaluation_core import run_policy_evaluation


SaveBestFn = Callable[[Path], None]
PolicyFactory = Callable[[], Callable]


def _atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(path)


def _read_existing_best(csv_path: Path) -> float:
    if not csv_path.exists():
        return float("inf")
    best = float("inf")
    try:
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    best = min(best, float(row["mean_cost"]))
                except Exception:
                    continue
    except Exception:
        pass
    return best


def _next_boundary(progress: int, interval: int) -> int:
    return ((max(int(progress), 0) // interval) + 1) * interval


class PeriodicEvaluator:
    """Evaluate the current policy on a fixed selection set during training.

    This object is intentionally algorithm-neutral. Trainers supply:
      * a `policy_factory` returning a deterministic policy_fn; and
      * a `save_best_fn(path_without_extension_or_full_path)` callback.

    Evaluation failures are visible but advisory: training continues.
    """

    CSV_FIELDS = [
        "eval_index",
        "progress_unit",
        "progress",
        "episode",
        "timestep",
        "mean_cost",
        "median_cost",
        "std_cost",
        "min_cost",
        "max_cost",
        "n_episodes",
        "eval_wall_time_s",
        "trainer_wall_time_s",
        "timestamp",
        "is_best",
    ]

    def __init__(
        self,
        *,
        technique: str,
        config: Dict[str, Any],
        tech_config: Dict[str, Any],
        student_config: Any,
        model_dir: Path,
        progress_unit: str,
        start_progress: int = 0,
        force: bool = False,
        trainer_start_wall: Optional[float] = None,
    ) -> None:
        self.technique = technique
        self.config = config
        self.tech_config = tech_config
        self.student_config = student_config
        self.model_dir = Path(model_dir)
        self.progress_unit = progress_unit
        self.trainer_start_wall = trainer_start_wall

        periodic = config.get("evaluation", {}).get("periodic", {})
        self.enabled = bool(periodic.get("enabled", True))
        self.seeds = list(periodic.get("seeds", [200, 201, 202, 203]))
        self.scenario_modes = list(
            periodic.get(
                "scenario_modes",
                config.get("evaluation", {}).get(
                    "scenario_modes",
                    ["stationary", "seasonal", "trend", "shock", "random"],
                ),
            )
        )

        if progress_unit == "steps":
            self.interval = int(
                tech_config.get(
                    "eval_interval_steps",
                    periodic.get("default_interval_steps", 250000),
                )
            )
        elif progress_unit == "episodes":
            self.interval = int(
                tech_config.get(
                    "eval_interval_episodes",
                    periodic.get("default_interval_episodes", 2000),
                )
            )
        else:
            raise ValueError(f"unsupported progress_unit: {progress_unit}")

        if self.interval <= 0:
            self.enabled = False

        self.csv_path = self.model_dir / "eval_metrics.csv"
        self.metadata_path = self.model_dir / "evaluation_metadata.json"
        self.details_dir = self.model_dir / "evaluations"

        if force:
            for path in (self.csv_path, self.metadata_path):
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
            if self.details_dir.exists():
                shutil.rmtree(self.details_dir)

        self.details_dir.mkdir(parents=True, exist_ok=True)
        self.best_mean_cost = _read_existing_best(self.csv_path)
        # A historical CSV without its selected checkpoint is insufficient for
        # resume-time model selection. Force the next successful evaluation to
        # recreate best_eval_model in that case.
        if not any(self.model_dir.glob("best_eval_model.*")):
            self.best_mean_cost = float("inf")
        self.eval_index = 0
        if self.csv_path.exists():
            try:
                with open(self.csv_path, newline="") as f:
                    self.eval_index = sum(1 for _ in csv.DictReader(f))
            except Exception:
                self.eval_index = 0

        self.next_progress = _next_boundary(start_progress, self.interval) \
            if self.enabled else math.inf
        self.last_evaluated_progress: Optional[int] = None

    def _append_csv(self, row: Dict[str, Any]) -> None:
        exists = self.csv_path.exists() and self.csv_path.stat().st_size > 0
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.CSV_FIELDS)
            if not exists:
                writer.writeheader()
            writer.writerow(row)

    def _write_metadata(self, row: Dict[str, Any]) -> None:
        payload = {
            "technique": self.technique,
            "enabled": self.enabled,
            "selection_seeds": self.seeds,
            "scenario_modes": self.scenario_modes,
            "episodes_per_evaluation": len(self.seeds) * len(self.scenario_modes),
            "progress_unit": self.progress_unit,
            "interval": self.interval,
            "evaluations_completed": self.eval_index,
            "best_mean_cost": (
                self.best_mean_cost if math.isfinite(self.best_mean_cost) else None
            ),
            "last_evaluation": row,
        }
        _atomic_json(self.metadata_path, payload)

    def _evaluate(
        self,
        *,
        progress: int,
        episode: Optional[int],
        policy_factory: PolicyFactory,
        save_best_fn: SaveBestFn,
    ) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None

        started = time.monotonic()
        try:
            policy_fn = policy_factory()
            result = run_policy_evaluation(
                policy_fn,
                self.student_config,
                self.seeds,
                self.scenario_modes,
            )
        except Exception as exc:
            print(
                f"  [eval] WARNING: deterministic evaluation failed at "
                f"{self.progress_unit}={int(progress):,}: {exc}"
            )
            return None

        eval_wall = time.monotonic() - started
        self.eval_index += 1
        is_best = result["mean_cost"] < self.best_mean_cost
        if is_best:
            try:
                save_best_fn(self.model_dir / "best_eval_model")
                self.best_mean_cost = result["mean_cost"]
            except Exception as exc:
                print(f"  [eval] WARNING: could not save best_eval_model: {exc}")
                is_best = False

        trainer_wall = None
        if self.trainer_start_wall is not None:
            trainer_wall = max(time.time() - self.trainer_start_wall, 0.0)

        row = {
            "eval_index": self.eval_index,
            "progress_unit": self.progress_unit,
            "progress": int(progress),
            "episode": int(episode) if episode is not None else None,
            "timestep": (
                int(progress) if self.progress_unit == "steps"
                else (int(episode) * 50 if episode is not None else None)
            ),
            "mean_cost": result["mean_cost"],
            "median_cost": result["median_cost"],
            "std_cost": result["std_cost"],
            "min_cost": result["min_cost"],
            "max_cost": result["max_cost"],
            "n_episodes": result["n_episodes"],
            "eval_wall_time_s": eval_wall,
            "trainer_wall_time_s": trainer_wall,
            "timestamp": datetime.now().isoformat(),
            "is_best": int(is_best),
        }
        self._append_csv(row)

        detail = {
            **row,
            "selection_seeds": self.seeds,
            "scenario_modes": self.scenario_modes,
            "records": result["records"],
        }
        _atomic_json(
            self.details_dir / f"eval_{self.eval_index:04d}_{int(progress)}.json",
            detail,
        )
        self._write_metadata(row)
        self.last_evaluated_progress = int(progress)

        marker = "  NEW BEST" if is_best else ""
        print(
            f"  [eval] {self.progress_unit}={int(progress):,} "
            f"eval_n={result['n_episodes']}  "
            f"mean={result['mean_cost']:,.0f}  "
            f"std={result['std_cost']:,.0f}  "
            f"time={eval_wall:.2f}s{marker}"
        )
        return row

    def maybe_evaluate(
        self,
        *,
        progress: int,
        episode: Optional[int],
        policy_factory: PolicyFactory,
        save_best_fn: SaveBestFn,
    ) -> Optional[Dict[str, Any]]:
        if not self.enabled or int(progress) < self.next_progress:
            return None
        result = self._evaluate(
            progress=int(progress),
            episode=episode,
            policy_factory=policy_factory,
            save_best_fn=save_best_fn,
        )
        while self.next_progress <= int(progress):
            self.next_progress += self.interval
        return result

    def evaluate_final(
        self,
        *,
        progress: int,
        episode: Optional[int],
        policy_factory: PolicyFactory,
        save_best_fn: SaveBestFn,
    ) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None
        if self.last_evaluated_progress == int(progress):
            return None
        return self._evaluate(
            progress=int(progress),
            episode=episode,
            policy_factory=policy_factory,
            save_best_fn=save_best_fn,
        )
