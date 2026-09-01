#!/usr/bin/env python3
"""Verify RunContext captures only diagnostics produced by this invocation."""

import csv
import json
import tempfile
from pathlib import Path

from run_context import RunContext


def write_diag(path, rows):
    fields = ["episode", "timestep", "wall_time_s", "timestamp", "epsilon"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        model_dir = root / "models" / "nn_sarsa"
        model_dir.mkdir(parents=True)
        (model_dir / "checkpoints").mkdir()
        (model_dir / "train_log.csv").write_text(
            "episode,timestep,episode_cost,wall_time_s,timestamp\n"
            "5,250,200.0,1.0,old\n", encoding="utf-8")
        (model_dir / "training_metadata.json").write_text(json.dumps({
            "steps_completed": 250, "episodes_completed": 5,
            "wall_time_s": 1.0, "best_rolling_cost": 200.0,
        }))
        write_diag(model_dir / "diagnostics.csv", [
            {"episode": 5, "timestep": 250, "wall_time_s": 1,
             "timestamp": "old", "epsilon": 0.9},
        ])

        tc = {"category": "nn_custom", "num_episodes": 7, "device": "auto"}
        full = {"techniques": {}, "diagnostics": {"enabled": True}}
        with RunContext(
            technique="nn_sarsa", tech_config=tc, full_config=full,
            model_dir=model_dir, force=False, run_name="phase3-synthetic",
            repo_root=root, runs_root=root / "runs",
        ) as ctx:
            (model_dir / "train_log.csv").write_text(
                "episode,timestep,episode_cost,wall_time_s,timestamp\n"
                "5,250,200.0,1.0,old\n"
                "6,300,180.0,1.2,new\n"
                "7,350,170.0,1.4,new\n", encoding="utf-8")
            (model_dir / "training_metadata.json").write_text(json.dumps({
                "steps_completed": 350, "episodes_completed": 7,
                "wall_time_s": 1.4, "best_rolling_cost": 170.0,
            }))
            write_diag(model_dir / "diagnostics.csv", [
                {"episode": 5, "timestep": 250, "wall_time_s": 1,
                 "timestamp": "old", "epsilon": 0.9},
                {"episode": 6, "timestep": 300, "wall_time_s": 1.2,
                 "timestamp": "new", "epsilon": 0.8},
                {"episode": 7, "timestep": 350, "wall_time_s": 1.4,
                 "timestamp": "new", "epsilon": 0.7},
            ])

        with (ctx.run_dir / "diagnostics.csv").open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert [r["episode"] for r in rows] == ["6", "7"]
        summary = json.loads((ctx.run_dir / "summary.json").read_text())
        manifest = json.loads((ctx.run_dir / "manifest.json").read_text())
        assert summary["diagnostic_rows"] == 2
        assert manifest["diagnostic_records"]["rows"] == 2
        assert manifest["result"]["diagnostic_rows"] == 2
        print("PASS: Phase-3 RunContext captures only current-run diagnostics")


if __name__ == "__main__":
    main()
