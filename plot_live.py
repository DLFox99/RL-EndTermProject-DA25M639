#!/usr/bin/env python3
"""Live training plotter — reads train_log.csv and refreshes.

Usage:
    python plot_live.py ppo          # watch one technique
    python plot_live.py all          # watch all side by side
    python plot_live.py ppo --save   # save PNG instead of live display
"""

import argparse
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).parent
MODELS_DIR = PROJECT_ROOT / "models"
PLOTS_DIR = PROJECT_ROOT / "plots"
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

ROLLING_WINDOW = 100


def load_train_log(tech_name):
    log_path = MODELS_DIR / tech_name / "train_log.csv"
    if not log_path.exists() or log_path.stat().st_size < 10:
        return None
    try:
        return pd.read_csv(log_path)
    except Exception:
        return None


def plot_techniques(techniques, save=False):
    if save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PLOTS_DIR.mkdir(exist_ok=True)
    n = len(techniques)

    if save:
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
        axes = axes[0]
        for i, tech in enumerate(techniques):
            df = load_train_log(tech)
            if df is not None and len(df) > 0:
                smoothed = df["episode_cost"].rolling(
                    ROLLING_WINDOW, min_periods=1).mean()
                axes[i].plot(df["episode"], smoothed, linewidth=0.8)
                axes[i].set_xlabel("Episode")
                axes[i].set_ylabel("Episode Cost")
            axes[i].set_title(tech.upper())
            axes[i].grid(True, alpha=0.3)
        plt.suptitle("Training Progress", y=1.02)
        plt.tight_layout()
        out = PLOTS_DIR / "training_progress.png"
        plt.savefig(str(out), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {out}")
        return

    # --- Live mode ---
    plt.ion()
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    axes = axes[0]

    print("Live plotter running. Close the window or Ctrl+C to stop.")
    try:
        while True:
            for i, tech in enumerate(techniques):
                axes[i].clear()
                df = load_train_log(tech)
                if df is not None and len(df) > 0:
                    smoothed = df["episode_cost"].rolling(
                        ROLLING_WINDOW, min_periods=1).mean()
                    axes[i].plot(df["episode"], smoothed, linewidth=0.8)
                    current = smoothed.iloc[-1]
                    best = smoothed.min()
                    axes[i].set_xlabel("Episode")
                    axes[i].set_ylabel("Episode Cost")
                    axes[i].set_title(
                        f"{tech.upper()}\ncurrent: {current:,.0f}  "
                        f"best: {best:,.0f}")
                else:
                    axes[i].set_title(f"{tech.upper()}\n(no data)")
                axes[i].grid(True, alpha=0.3)

            plt.suptitle(
                f"Training Progress  "
                f"(refreshing every 30s)", y=1.02)
            plt.tight_layout()
            plt.pause(30)

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        plt.close("all")


def main():
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    all_techniques = list(config["techniques"].keys())

    parser = argparse.ArgumentParser(description="Plot training progress")
    parser.add_argument("technique", choices=all_techniques + ["all"])
    parser.add_argument("--save", action="store_true",
                        help="Save PNG instead of live display")
    args = parser.parse_args()

    techniques = all_techniques if args.technique == "all" else [args.technique]
    plot_techniques(techniques, save=args.save)


if __name__ == "__main__":
    main()
