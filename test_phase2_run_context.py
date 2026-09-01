#!/usr/bin/env python3
"""Verify Phase-2 eval records are captured by the immutable RunContext."""

import csv
import json
import tempfile
from pathlib import Path

from run_context import RunContext


def main():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        model_dir = root / "models" / "nn_sarsa"
        model_dir.mkdir(parents=True)
        (model_dir / "checkpoints").mkdir()

        full = {"techniques": {}, "evaluation": {"periodic": {"enabled": True}}}
        tc = {"category": "nn_custom", "num_episodes": 10, "device": "auto"}

        with RunContext(
            technique="nn_sarsa",
            tech_config=tc,
            full_config=full,
            model_dir=model_dir,
            force=True,
            run_name="phase2-synthetic",
            repo_root=root,
            runs_root=root / "runs",
        ) as ctx:
            (model_dir / "train_log.csv").write_text(
                "episode,timestep,episode_cost,wall_time_s,timestamp\n"
                "10,500,100.0,1.0,now\n"
            )
            (model_dir / "training_metadata.json").write_text(json.dumps({
                "steps_completed": 500,
                "episodes_completed": 10,
                "wall_time_s": 1.0,
                "best_rolling_cost": 100.0,
            }))
            with (model_dir / "eval_metrics.csv").open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["eval_index", "mean_cost"])
                w.writeheader(); w.writerow({"eval_index": 1, "mean_cost": 88.0})
            (model_dir / "evaluation_metadata.json").write_text(json.dumps({
                "best_mean_cost": 88.0,
                "evaluations_completed": 1,
                "episodes_per_evaluation": 20,
            }))
            evdir = model_dir / "evaluations"; evdir.mkdir()
            (evdir / "eval_0001_10.json").write_text('{"mean_cost":88.0}')
            (model_dir / "best_eval_model.pt").write_bytes(b"eval-best")
            (model_dir / "final_model.pt").write_bytes(b"final")

        summary = json.loads((ctx.run_dir / "summary.json").read_text())
        manifest = json.loads((ctx.run_dir / "manifest.json").read_text())
        assert summary["best_eval_mean_cost"] == 88.0
        assert summary["evaluations_completed"] == 1
        assert (ctx.run_dir / "eval_metrics.csv").exists()
        assert (ctx.run_dir / "evaluations" / "eval_0001_10.json").exists()
        assert (ctx.run_dir / "artifacts" / "best_eval_model.pt").exists()
        assert manifest["evaluation_records"]["rows"] == 1
        print("PASS: Phase-2 RunContext evaluation capture test")


if __name__ == "__main__":
    main()
