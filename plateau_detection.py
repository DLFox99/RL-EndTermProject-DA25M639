"""Deterministic-evaluation plateau detection (advisory only).

The detector operates on periodic fixed-seed evaluation means.  It deliberately
never changes a training budget; it only records when meaningful improvement
has stopped for a configurable number of evaluation points.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from math import isfinite
from typing import Any, Dict, Iterable, Optional


@dataclass
class PlateauState:
    evaluations_seen: int = 0
    reference_best_cost: Optional[float] = None
    best_observed_cost: Optional[float] = None
    no_improvement_count: int = 0
    detected: bool = False
    first_detected_progress: Optional[int] = None
    reason: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EvalPlateauDetector:
    """Sticky plateau detector using deterministic evaluation means.

    A new evaluation is considered a *significant* improvement only if it beats
    the current reference best by at least

        max(min_improvement_abs,
            abs(reference_best) * min_improvement_frac)

    Small changes below that threshold do not reset patience.  Detection is
    sticky and advisory: callers should keep training unless a separate sweep
    controller intentionally uses the signal.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        min_evaluations: int = 5,
        patience: int = 4,
        min_improvement_abs: float = 500.0,
        min_improvement_frac: float = 0.005,
    ) -> None:
        self.enabled = bool(enabled)
        self.min_evaluations = max(int(min_evaluations), 1)
        self.patience = max(int(patience), 1)
        self.min_improvement_abs = max(float(min_improvement_abs), 0.0)
        self.min_improvement_frac = max(float(min_improvement_frac), 0.0)
        self.state = PlateauState()

    def threshold(self) -> float:
        ref = self.state.reference_best_cost
        if ref is None or not isfinite(ref):
            return self.min_improvement_abs
        return max(
            self.min_improvement_abs,
            abs(float(ref)) * self.min_improvement_frac,
        )

    def observe(self, mean_cost: float, progress: int) -> Dict[str, Any]:
        value = float(mean_cost)
        progress = int(progress)
        if not isfinite(value):
            return {
                "improvement_abs": 0.0,
                "improvement_frac": 0.0,
                "significant_improvement": 0,
                "no_improvement_count": self.state.no_improvement_count,
                "plateau_detected": int(self.state.detected),
                "plateau_reason": self.state.reason or "",
            }

        self.state.evaluations_seen += 1
        if self.state.best_observed_cost is None:
            self.state.best_observed_cost = value
        else:
            self.state.best_observed_cost = min(self.state.best_observed_cost, value)

        previous_ref = self.state.reference_best_cost
        if previous_ref is None:
            improvement_abs = 0.0
            improvement_frac = 0.0
            significant = True
            self.state.reference_best_cost = value
            self.state.no_improvement_count = 0
        else:
            improvement_abs = float(previous_ref - value)
            denom = max(abs(float(previous_ref)), 1e-12)
            improvement_frac = improvement_abs / denom
            significant = improvement_abs >= self.threshold()
            if significant:
                self.state.reference_best_cost = value
                self.state.no_improvement_count = 0
            else:
                self.state.no_improvement_count += 1

        if (
            self.enabled
            and not self.state.detected
            and self.state.evaluations_seen >= self.min_evaluations
            and self.state.no_improvement_count >= self.patience
        ):
            self.state.detected = True
            self.state.first_detected_progress = progress
            self.state.reason = (
                f"no significant deterministic-eval improvement for "
                f"{self.state.no_improvement_count} evaluations "
                f"(threshold=max({self.min_improvement_abs:g}, "
                f"{self.min_improvement_frac:.3%} of reference best))"
            )

        return {
            "improvement_abs": float(improvement_abs),
            "improvement_frac": float(improvement_frac),
            "significant_improvement": int(bool(significant)),
            "no_improvement_count": int(self.state.no_improvement_count),
            "plateau_detected": int(self.state.detected),
            "plateau_reason": self.state.reason or "",
        }

    def replay(self, rows: Iterable[Dict[str, Any]]) -> None:
        """Rebuild detector state from historical evaluation rows."""
        for row in rows:
            try:
                mean = float(row["mean_cost"])
                progress = int(float(row["progress"]))
            except Exception:
                continue
            self.observe(mean, progress)
