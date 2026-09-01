#!/usr/bin/env python3
"""Fast synthetic test for run_context.py; does not import the RL environment."""

import json
import tempfile
import time
from pathlib import Path

from run_context import RunContext


def main():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        model_dir = root / "models" / "nn_sarsa"
        model_dir.mkdir(parents=True)
        (model_dir / "checkpoints").mkdir()

        full = {"checkpoint_interval_min": 5, "techniques": {}}
        tc = {
            "category": "nn_custom",
            "portal_name": "Neural Network based SARSA",
            "num_episodes": 4,
            "device": "auto",
        }

        with RunContext(
            technique="nn_sarsa",
            tech_config=tc,
            full_config=full,
            model_dir=model_dir,
            force=True,
            run_name="synthetic",
            repo_root=root,
            runs_root=root / "runs",
        ) as ctx:
            log = model_dir / "train_log.csv"
            log.write_text(
                "episode,timestep,episode_cost,wall_time_s,timestamp\n",
                encoding="utf-8",
            )
            with log.open("a", encoding="utf-8") as f:
                for ep, cost in enumerate([400.0, 300.0, 250.0, 200.0], start=1):
                    f.write(f"{ep},{ep*50},{cost},{ep*0.01},2026-09-01T00:00:00\n")
            (model_dir / "training_metadata.json").write_text(
                json.dumps({
                    "steps_completed": 200,
                    "episodes_completed": 4,
                    "wall_time_s": 0.04,
                    "best_rolling_cost": 200.0,
                }),
                encoding="utf-8",
            )
            (model_dir / "best_model.pt").write_bytes(b"best")
            (model_dir / "final_model.pt").write_bytes(b"final")
            time.sleep(0.05)

        manifest = json.loads((ctx.run_dir / "manifest.json").read_text())
        summary = json.loads((ctx.run_dir / "summary.json").read_text())
        assert manifest["status"] == "completed", manifest
        assert summary["completed_episodes"] == 4, summary
        assert summary["completed_steps"] == 200, summary
        assert (ctx.run_dir / "train_metrics.csv").exists()
        assert (ctx.run_dir / "resolved_config.yaml").exists()
        assert (ctx.run_dir / "artifacts" / "best_model.pt").exists()
        assert (ctx.run_dir / "artifacts" / "final_model.pt").exists()
        assert manifest["artifacts"]["best_model.pt"]["sha256"]
        print("PASS: synthetic RunContext test")
        print(ctx.run_dir)


if __name__ == "__main__":
    main()
