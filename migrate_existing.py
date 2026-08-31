#!/usr/bin/env python3
"""Migrate existing trained models into the pipeline directory structure.

Run once after setting up the pipeline. Handles both flat models/ layout
(ppo_model.zip in models/) and already-structured layout (models/ppo/).

Usage:
    python migrate_existing.py                     # auto-detect
    python migrate_existing.py --steps 5000000     # set steps completed
"""

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

MODELS_DIR = Path(__file__).parent / "models"

OLD_NAMES = {
    "ppo":       ("ppo_model.zip",       "final_model.zip"),
    "a2c":       ("a2c_model.zip",       "final_model.zip"),
    "dqn":       ("dqn_model.zip",       "final_model.zip"),
    "ddqn":      ("ddqn_model.zip",      "final_model.zip"),
    "reinforce": ("reinforce_model.pt",  "final_model.pt"),
}


def migrate(steps_completed):
    print(f"Migrating models (assuming {steps_completed:,} steps completed)...\n")

    for tech, (old_name, new_name) in OLD_NAMES.items():
        tech_dir = MODELS_DIR / tech
        tech_dir.mkdir(parents=True, exist_ok=True)
        (tech_dir / "checkpoints").mkdir(exist_ok=True)

        dst = tech_dir / new_name

        # Check if already migrated
        if dst.exists():
            print(f"  {tech}: already has {new_name}, skipping")
            continue

        # Check for flat file in models/
        flat_src = MODELS_DIR / old_name
        if flat_src.exists():
            shutil.move(str(flat_src), str(dst))
            print(f"  {tech}: moved {old_name} → {tech}/{new_name}")
        else:
            print(f"  {tech}: {old_name} not found, skipping")
            continue

        # Create metadata
        episodes = steps_completed // 50
        meta = {
            "steps_completed": steps_completed,
            "episodes_completed": episodes,
            "wall_time_s": 0,
            "best_rolling_cost": None,
            "migrated_from": old_name,
            "migrated_at": datetime.now().isoformat(),
        }
        with open(tech_dir / "training_metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        # Empty train log (we don't have historical data)
        log_path = tech_dir / "train_log.csv"
        if not log_path.exists():
            with open(log_path, "w") as f:
                f.write("episode,timestep,episode_cost,wall_time_s,timestamp\n")

        print(f"  {tech}: metadata created (steps={steps_completed:,})")

    print("\nMigration complete.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=5000000,
                        help="Steps completed for existing models")
    args = parser.parse_args()
    migrate(args.steps)


if __name__ == "__main__":
    main()
