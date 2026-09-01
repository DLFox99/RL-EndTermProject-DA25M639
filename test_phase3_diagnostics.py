#!/usr/bin/env python3
"""Dependency-light Phase-3 diagnostics/layout regression tests."""

import csv
import tempfile
from pathlib import Path

import diagnostics


def main():
    expected_grids = {
        1: (1, 1), 2: (1, 2), 3: (2, 2), 4: (2, 2),
        5: (2, 3), 6: (2, 3), 7: (2, 4), 8: (2, 4), 9: (3, 3),
    }
    for n, expected in expected_grids.items():
        assert diagnostics.subplot_grid(n) == expected, (n, diagnostics.subplot_grid(n))

    expected_panels = {
        "ppo": 8, "a2c": 6, "dqn": 6, "ddqn": 6,
        "nn_qlearning": 8, "nn_sarsa": 8, "reinforce": 6,
        "a3c": 8, "tabular_qlearning": 6, "tabular_sarsa": 6,
        "td_lambda": 8,
    }
    for tech, n in expected_panels.items():
        assert len(diagnostics.PANEL_SPECS[tech]) == n
        rows, cols = diagnostics.subplot_grid(n)
        assert rows * cols >= n
        assert rows > 1 or cols <= 2  # no long single-row layout
        assert cols > 1 or rows <= 1  # no long single-column layout
        assert diagnostics.DIAGNOSTIC_COLUMNS[tech][:4] == diagnostics.COMMON_COLUMNS

    mirrored = []
    old_log = diagnostics.wandb_utils.log
    diagnostics.wandb_utils.log = lambda data, step=None: mirrored.append((data, step))
    try:
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            logger = diagnostics.DiagnosticLogger(d, "nn_sarsa", force=True)
            logger.log({"epsilon": 0.5, "loss_mean": 2.25}, episode=3,
                       timestep=150, wall_time_s=1.2)
            logger.log({"epsilon": 0.4, "loss_mean": 1.75}, episode=4,
                       timestep=200, wall_time_s=1.5)
            with (d / "diagnostics.csv").open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            assert len(rows) == 2
            assert rows[0]["episode"] == "3" and rows[1]["timestep"] == "200"
            assert float(rows[0]["epsilon"]) == 0.5
            assert len(mirrored) == 2
            assert "diagnostics/epsilon" in mirrored[-1][0]
    finally:
        diagnostics.wandb_utils.log = old_log

    print("PASS: Phase-3 diagnostics schemas, W&B mirror, and balanced subplot grids")


if __name__ == "__main__":
    main()
