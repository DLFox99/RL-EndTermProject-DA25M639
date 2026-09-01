#!/usr/bin/env python3
"""Algorithm-specific diagnostics for Phase 3.

Local CSV is authoritative. W&B receives the same metrics only through the
existing throttled wandb_utils.log() mirror.

This module intentionally contains no learning-rule logic.
"""

from __future__ import annotations

import csv
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

import wandb_utils


COMMON_COLUMNS = [
    "episode",
    "timestep",
    "wall_time_s",
    "timestamp",
]

DIAGNOSTIC_COLUMNS: Dict[str, List[str]] = {
    "ppo": COMMON_COLUMNS + [
        "policy_gradient_loss",
        "value_loss",
        "entropy_loss",
        "approx_kl",
        "clip_fraction",
        "explained_variance",
        "learning_rate",
        "n_updates",
    ],
    "a2c": COMMON_COLUMNS + [
        "policy_loss",
        "value_loss",
        "entropy_loss",
        "explained_variance",
        "learning_rate",
        "n_updates",
    ],
    "dqn": COMMON_COLUMNS + [
        "loss",
        "exploration_rate",
        "learning_rate",
        "n_updates",
        "replay_size",
        "replay_fraction",
        "q_mean",
        "q_std",
        "q_min",
        "q_max",
        "target_q_mean",
        "target_q_std",
    ],
    "ddqn": COMMON_COLUMNS + [
        "loss",
        "exploration_rate",
        "learning_rate",
        "n_updates",
        "replay_size",
        "replay_fraction",
        "q_mean",
        "q_std",
        "q_min",
        "q_max",
        "target_q_mean",
        "target_q_std",
    ],
    "nn_qlearning": COMMON_COLUMNS + [
        "epsilon",
        "loss_mean",
        "td_error_abs_mean",
        "td_error_abs_max",
        "selected_q_mean",
        "selected_q_std",
        "target_mean",
        "target_std",
        "grad_norm_mean",
        "grad_norm_max",
        "q_min",
        "q_max",
        "q_abs_max",
    ],
    "nn_sarsa": COMMON_COLUMNS + [
        "epsilon",
        "loss_mean",
        "td_error_abs_mean",
        "td_error_abs_max",
        "selected_q_mean",
        "selected_q_std",
        "target_mean",
        "target_std",
        "grad_norm_mean",
        "grad_norm_max",
        "q_min",
        "q_max",
        "q_abs_max",
    ],
    "reinforce": COMMON_COLUMNS + [
        "policy_loss",
        "baseline",
        "advantage_mean",
        "advantage_std",
        "entropy",
        "grad_norm",
    ],
    "a3c": COMMON_COLUMNS + [
        "worker_id",
        "policy_loss",
        "value_loss",
        "entropy",
        "total_loss",
        "advantage_mean",
        "advantage_std",
        "grad_norm",
        "worker_return",
        "worker_return_roll_mean",
        "worker_return_roll_std",
        "worker_cost_spread",
    ],
    "tabular_qlearning": COMMON_COLUMNS + [
        "epsilon",
        "td_error_abs_mean",
        "td_error_abs_max",
        "q_update_abs_mean",
        "q_update_abs_max",
        "q_touched_min",
        "q_touched_max",
        "q_abs_max",
        "state_coverage",
    ],
    "tabular_sarsa": COMMON_COLUMNS + [
        "epsilon",
        "td_error_abs_mean",
        "td_error_abs_max",
        "q_update_abs_mean",
        "q_update_abs_max",
        "q_touched_min",
        "q_touched_max",
        "q_abs_max",
        "state_coverage",
    ],
    "td_lambda": COMMON_COLUMNS + [
        "epsilon",
        "td_error_abs_mean",
        "td_error_abs_max",
        "q_update_abs_mean",
        "q_update_abs_max",
        "q_touched_min",
        "q_touched_max",
        "q_abs_max",
        "state_coverage",
        "active_trace_mean",
        "active_trace_max",
        "trace_abs_mean",
        "trace_abs_max",
    ],
}


# Each tuple: (panel title, [metric columns]).
# Cost + deterministic evaluation is handled specially by plot_live.py.
PANEL_SPECS: Dict[str, List[Tuple[str, List[str]]]] = {
    "ppo": [
        ("Cost + Evaluation", []),
        ("Policy Gradient Loss", ["policy_gradient_loss"]),
        ("Value Loss", ["value_loss"]),
        ("Entropy Loss", ["entropy_loss"]),
        ("Approx KL", ["approx_kl"]),
        ("Clip Fraction", ["clip_fraction"]),
        ("Explained Variance", ["explained_variance"]),
        ("Learning Rate", ["learning_rate"]),
    ],
    "a2c": [
        ("Cost + Evaluation", []),
        ("Policy Loss", ["policy_loss"]),
        ("Value Loss", ["value_loss"]),
        ("Entropy Loss", ["entropy_loss"]),
        ("Explained Variance", ["explained_variance"]),
        ("Learning Rate", ["learning_rate"]),
    ],
    "dqn": [
        ("Cost + Evaluation", []),
        ("TD / Training Loss", ["loss"]),
        ("Q Values", ["q_mean", "q_std"]),
        ("TD Target", ["target_q_mean", "target_q_std"]),
        ("Exploration Rate", ["exploration_rate"]),
        ("Replay Buffer Fill", ["replay_fraction"]),
    ],
    "ddqn": [
        ("Cost + Evaluation", []),
        ("TD / Training Loss", ["loss"]),
        ("Q Values", ["q_mean", "q_std"]),
        ("TD Target", ["target_q_mean", "target_q_std"]),
        ("Exploration Rate", ["exploration_rate"]),
        ("Replay Buffer Fill", ["replay_fraction"]),
    ],
    "nn_qlearning": [
        ("Cost + Evaluation", []),
        ("Training Loss", ["loss_mean"]),
        ("|TD Error|", ["td_error_abs_mean", "td_error_abs_max"]),
        ("Selected Q", ["selected_q_mean", "selected_q_std"]),
        ("TD Target", ["target_mean", "target_std"]),
        ("Pre-clip Gradient Norm", ["grad_norm_mean", "grad_norm_max"]),
        ("Epsilon", ["epsilon"]),
        ("Q Range", ["q_min", "q_max"]),
    ],
    "nn_sarsa": [
        ("Cost + Evaluation", []),
        ("Training Loss", ["loss_mean"]),
        ("|TD Error|", ["td_error_abs_mean", "td_error_abs_max"]),
        ("Selected Q", ["selected_q_mean", "selected_q_std"]),
        ("SARSA Target", ["target_mean", "target_std"]),
        ("Pre-clip Gradient Norm", ["grad_norm_mean", "grad_norm_max"]),
        ("Epsilon", ["epsilon"]),
        ("Q Range", ["q_min", "q_max"]),
    ],
    "reinforce": [
        ("Cost + Evaluation", []),
        ("Policy Loss", ["policy_loss"]),
        ("Baseline", ["baseline"]),
        ("Advantage", ["advantage_mean", "advantage_std"]),
        ("Policy Entropy", ["entropy"]),
        ("Pre-clip Gradient Norm", ["grad_norm"]),
    ],
    "a3c": [
        ("Cost + Evaluation", []),
        ("Policy Loss", ["policy_loss"]),
        ("Value Loss", ["value_loss"]),
        ("Entropy", ["entropy"]),
        ("Advantage", ["advantage_mean", "advantage_std"]),
        ("Pre-clip Gradient Norm", ["grad_norm"]),
        ("Worker Return", ["worker_return_roll_mean", "worker_return_roll_std"]),
        ("Worker Cost Spread", ["worker_cost_spread"]),
    ],
    "tabular_qlearning": [
        ("Cost + Evaluation", []),
        ("Epsilon", ["epsilon"]),
        ("|TD Error|", ["td_error_abs_mean", "td_error_abs_max"]),
        ("Q Update Magnitude", ["q_update_abs_mean", "q_update_abs_max"]),
        ("Touched Q Range", ["q_touched_min", "q_touched_max"]),
        ("State Visitation", ["state_coverage"]),
    ],
    "tabular_sarsa": [
        ("Cost + Evaluation", []),
        ("Epsilon", ["epsilon"]),
        ("|TD Error|", ["td_error_abs_mean", "td_error_abs_max"]),
        ("Q Update Magnitude", ["q_update_abs_mean", "q_update_abs_max"]),
        ("Touched Q Range", ["q_touched_min", "q_touched_max"]),
        ("State Visitation", ["state_coverage"]),
    ],
    "td_lambda": [
        ("Cost + Evaluation", []),
        ("Epsilon", ["epsilon"]),
        ("|TD Error|", ["td_error_abs_mean", "td_error_abs_max"]),
        ("Q Update Magnitude", ["q_update_abs_mean", "q_update_abs_max"]),
        ("Touched Q Range", ["q_touched_min", "q_touched_max"]),
        ("Active Eligibility Traces", ["active_trace_mean", "active_trace_max"]),
        ("Eligibility Magnitude", ["trace_abs_mean", "trace_abs_max"]),
        ("State Visitation", ["state_coverage"]),
    ],
}


EPISODE_X_TECHNIQUES = {
    "reinforce", "a3c",
    "tabular_qlearning", "tabular_sarsa", "td_lambda",
    "nn_qlearning", "nn_sarsa",
}


def subplot_grid(n_panels: int) -> Tuple[int, int]:
    """Balanced fixed layouts agreed for Phase 3."""
    n = max(int(n_panels), 1)
    if n == 1:
        return 1, 1
    if n == 2:
        return 1, 2
    if n <= 4:
        return 2, 2
    if n <= 6:
        return 2, 3
    if n <= 8:
        return 2, 4
    if n <= 9:
        return 3, 3
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    return rows, cols


def x_column_for(technique: str) -> str:
    return "episode" if technique in EPISODE_X_TECHNIQUES else "timestep"


def _safe_scalar(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    try:
        # Torch scalar without importing torch.
        if hasattr(value, "detach") and hasattr(value, "item"):
            return value.detach().item()
    except Exception:
        pass
    return value


class DiagnosticLogger:
    """Append fixed-schema algorithm diagnostics to models/<tech>/diagnostics.csv."""

    def __init__(
        self,
        model_dir: Path,
        technique: str,
        *,
        enabled: bool = True,
        force: bool = False,
        start_wall: Optional[float] = None,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.technique = str(technique)
        self.enabled = bool(enabled)
        self.path = self.model_dir / "diagnostics.csv"
        self.start_wall = start_wall
        self.columns = DIAGNOSTIC_COLUMNS.get(self.technique, COMMON_COLUMNS)

        if self.enabled:
            self.model_dir.mkdir(parents=True, exist_ok=True)
            if force and self.path.exists():
                self.path.unlink()
            if not self.path.exists() or self.path.stat().st_size == 0:
                with self.path.open("w", newline="", encoding="utf-8") as f:
                    csv.DictWriter(f, fieldnames=self.columns).writeheader()

    def log(
        self,
        metrics: Mapping[str, Any],
        *,
        episode: Optional[int] = None,
        timestep: Optional[int] = None,
        wall_time_s: Optional[float] = None,
    ) -> None:
        if not self.enabled:
            return

        row: Dict[str, Any] = {key: "" for key in self.columns}
        if episode is not None:
            row["episode"] = int(episode)
        if timestep is not None:
            row["timestep"] = int(timestep)
        if wall_time_s is not None:
            row["wall_time_s"] = float(wall_time_s)
        row["timestamp"] = datetime.now().isoformat()

        for key, value in metrics.items():
            if key in row:
                row[key] = _safe_scalar(value)

        try:
            with self.path.open("a", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=self.columns).writerow(row)
        except Exception as exc:
            print(f"  [diagnostics] write failed: {exc}")
            return

        # Optional online mirror. Local CSV above remains authoritative and
        # wandb_utils itself enforces the existing ~5-second emission throttle.
        wb = {
            f"diagnostics/{key}": value
            for key, value in row.items()
            if key not in {"timestamp", "episode", "timestep", "wall_time_s", "worker_id"}
            and value not in ("", None)
        }
        if wb:
            try:
                wandb_utils.log(wb, step=timestep)
            except Exception:
                pass


def read_diagnostics(path: Path):
    """Small helper used by tests/plotter without imposing pandas here."""
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))
