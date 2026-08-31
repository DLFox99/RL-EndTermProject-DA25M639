#!/usr/bin/env python3
"""Log leaderboard results for tracking improvement over time.

Usage:
    python leaderboard_log.py --cost 268856 --notes "initial 5 techniques"
    python leaderboard_log.py --show
"""

import argparse
import csv
from datetime import datetime
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).parent
LOG_PATH = PROJECT_ROOT / "leaderboard.csv"
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
RESULTS_DIR = PROJECT_ROOT / "results"

HEADER = ["timestamp", "overall_cost", "local_best5_avg",
          "techniques_uploaded", "change_made", "notes"]


def _ensure_file():
    if not LOG_PATH.exists():
        with open(LOG_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(HEADER)


def _get_local_best5():
    """Read comparison.csv and return avg of best 5."""
    comp_path = RESULTS_DIR / "comparison.csv"
    if not comp_path.exists():
        return None
    import pandas as pd
    df = pd.read_csv(comp_path)
    best5 = df.nsmallest(5, "Avg Cost")
    return best5["Avg Cost"].mean()


def _get_techniques_list():
    """List techniques that have models."""
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    models_dir = PROJECT_ROOT / "models"
    trained = []
    for tech in config["techniques"]:
        td = models_dir / tech
        if (td / "final_model.zip").exists() or (td / "final_model.pt").exists() \
                or (td / "best_model.zip").exists() or (td / "best_model.pt").exists():
            trained.append(tech)
    return ",".join(trained)


def log_entry(cost, change="", notes=""):
    _ensure_file()
    local_avg = _get_local_best5()
    techs = _get_techniques_list()

    row = [
        datetime.now().isoformat(timespec="seconds"),
        f"{cost:.2f}",
        f"{local_avg:.2f}" if local_avg else "",
        techs,
        change,
        notes,
    ]
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)
    print(f"Logged: cost={cost:.0f}  local_best5={local_avg or 'N/A'}")


def show():
    _ensure_file()
    with open(LOG_PATH) as f:
        content = f.read()
    print(content)


def main():
    parser = argparse.ArgumentParser(description="Log leaderboard results")
    parser.add_argument("--cost", type=float, help="Overall leaderboard cost")
    parser.add_argument("--change", type=str, default="",
                        help="What changed since last entry")
    parser.add_argument("--notes", type=str, default="")
    parser.add_argument("--show", action="store_true",
                        help="Show leaderboard history")
    args = parser.parse_args()

    if args.show:
        show()
    elif args.cost is not None:
        log_entry(args.cost, args.change, args.notes)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
