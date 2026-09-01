#!/usr/bin/env python3
"""Phase-4 advisory deterministic-evaluation plateau tests."""

import csv
import tempfile
from pathlib import Path

from plateau_detection import EvalPlateauDetector
from periodic_evaluation import PeriodicEvaluator


def test_detector():
    d = EvalPlateauDetector(
        enabled=True,
        min_evaluations=5,
        patience=3,
        min_improvement_abs=100.0,
        min_improvement_frac=0.0,
    )
    # Meaningful improvements reset patience.
    for i, value in enumerate([1000, 850, 700], 1):
        row = d.observe(value, i * 100)
        assert row["plateau_detected"] == 0
        assert row["no_improvement_count"] == 0

    # Three sub-threshold evaluations after enough observations -> plateau.
    d.observe(660, 400)   # +40 only
    d.observe(640, 500)   # +60 from reference 700
    row = d.observe(630, 600)  # +70 from reference 700
    assert row["plateau_detected"] == 1
    assert d.state.first_detected_progress == 600
    assert "3 evaluations" in d.state.reason

    # Sticky even if a later meaningful improvement arrives.
    d.observe(400, 700)
    assert d.state.detected is True
    assert d.state.first_detected_progress == 600


def test_eval_csv_upgrade_and_replay():
    old_fields = [
        "eval_index", "progress_unit", "progress", "episode", "timestep",
        "mean_cost", "median_cost", "std_cost", "min_cost", "max_cost",
        "n_episodes", "eval_wall_time_s", "trainer_wall_time_s",
        "timestamp", "is_best",
    ]
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        model_dir = root / "model"
        model_dir.mkdir()
        with (model_dir / "eval_metrics.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=old_fields)
            w.writeheader()
            w.writerow({
                "eval_index": 1, "progress_unit": "episodes", "progress": 100,
                "episode": 100, "timestep": 5000, "mean_cost": 1000,
                "median_cost": 1000, "std_cost": 10, "min_cost": 990,
                "max_cost": 1010, "n_episodes": 20, "eval_wall_time_s": 1,
                "trainer_wall_time_s": 10, "timestamp": "x", "is_best": 1,
            })
        # Pretend the historical selected checkpoint exists.
        (model_dir / "best_eval_model.pt").write_bytes(b"x")
        cfg = {
            "evaluation": {
                "periodic": {
                    "enabled": True,
                    "seeds": [1],
                    "scenario_modes": ["random"],
                    "default_interval_episodes": 100,
                },
                "plateau": {
                    "enabled": True,
                    "min_evaluations": 3,
                    "patience": 2,
                    "min_improvement_abs": 10,
                    "min_improvement_frac": 0,
                },
            }
        }
        pe = PeriodicEvaluator(
            technique="x", config=cfg, tech_config={}, student_config=None,
            model_dir=model_dir, progress_unit="episodes", start_progress=100,
            force=False,
        )
        with (model_dir / "eval_metrics.csv").open(newline="") as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == PeriodicEvaluator.CSV_FIELDS
            rows = list(reader)
        assert len(rows) == 1
        assert pe.eval_index == 1
        assert pe.plateau_detector.state.evaluations_seen == 1
        assert pe.best_mean_cost == 1000.0


if __name__ == "__main__":
    test_detector()
    test_eval_csv_upgrade_and_replay()
    print("PASS: Phase-4 advisory deterministic-evaluation plateau detection")
