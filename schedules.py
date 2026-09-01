"""Small dependency-free scalar schedules used by controlled experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple


@dataclass(frozen=True)
class ScalarSchedule:
    kind: str
    start: float
    end: float
    duration: int
    points: Tuple[Tuple[int, float], ...] = ()

    def value(self, progress: int) -> float:
        p = max(int(progress), 0)
        kind = self.kind.lower()
        if kind == "constant":
            return float(self.start)
        if kind == "piecewise":
            return self._piecewise(p)

        d = max(int(self.duration), 1)
        frac = min(max(p / float(d), 0.0), 1.0)
        if kind == "linear":
            return self.start + (self.end - self.start) * frac
        if kind == "cosine":
            weight = 0.5 * (1.0 - math.cos(math.pi * frac))
            return self.start + (self.end - self.start) * weight
        if kind == "exponential":
            if self.start <= 0 or self.end <= 0:
                raise ValueError("exponential schedule requires start/end > 0")
            return self.start * ((self.end / self.start) ** frac)
        raise ValueError(f"unsupported schedule type: {self.kind}")

    def _piecewise(self, progress: int) -> float:
        pts = list(self.points)
        if not pts:
            return float(self.start)
        if progress <= pts[0][0]:
            return float(pts[0][1])
        for (p0, v0), (p1, v1) in zip(pts, pts[1:]):
            if progress <= p1:
                if p1 == p0:
                    return float(v1)
                frac = (progress - p0) / float(p1 - p0)
                return float(v0 + (v1 - v0) * frac)
        return float(pts[-1][1])


def schedule_from_spec(spec: Dict[str, Any] | None) -> ScalarSchedule | None:
    """Build a schedule from config; ``None`` means preserve legacy behavior."""
    if not spec:
        return None
    kind = str(spec.get("type", "linear")).lower()
    if kind == "piecewise":
        raw_points = spec.get("points", [])
        points: List[Tuple[int, float]] = []
        for item in raw_points:
            if isinstance(item, dict):
                points.append((int(item["progress"]), float(item["value"])))
            else:
                progress, value = item
                points.append((int(progress), float(value)))
        points.sort(key=lambda x: x[0])
        if not points:
            raise ValueError("piecewise schedule requires at least one point")
        return ScalarSchedule(
            kind="piecewise",
            start=float(points[0][1]),
            end=float(points[-1][1]),
            duration=max(points[-1][0], 1),
            points=tuple(points),
        )

    start = float(spec["start"])
    end = float(spec.get("end", start))
    duration = max(int(spec.get("duration", 1)), 1)
    return ScalarSchedule(kind=kind, start=start, end=end, duration=duration)


def configured_schedule(tech_config: Dict[str, Any], name: str) -> ScalarSchedule | None:
    schedules = tech_config.get("schedules", {})
    if not isinstance(schedules, dict):
        raise ValueError("technique 'schedules' must be a mapping")
    spec = schedules.get(name)
    if spec is None:
        return None
    if not isinstance(spec, dict):
        raise ValueError(f"schedule {name!r} must be a mapping")
    return schedule_from_spec(spec)


def set_optimizer_lr(optimizer: Any, value: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(value)


def sb3_progress_schedule(schedule: ScalarSchedule, total_progress: int):
    """Convert a progress-based schedule to the SB3 progress_remaining API."""
    total = max(int(total_progress), 1)

    def fn(progress_remaining: float) -> float:
        elapsed = int(round((1.0 - float(progress_remaining)) * total))
        return float(schedule.value(elapsed))

    return fn
