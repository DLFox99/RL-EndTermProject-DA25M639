#!/usr/bin/env python3
"""Live training plotter — reads train_log.csv and refreshes.

Defaults to log-log scale: timesteps commonly span 3+ orders of magnitude
(early rollout to millions of steps) and cost typically spans ~1 order of
magnitude (steep initial drop, then plateau). Log-log gives both the early
steep-descent phase and the later plateau comparable visual detail, instead
of compressing the interesting part into a sliver on a linear axis.

Handles discontinuous training history correctly: if a technique was
migrated from a pre-trained model (e.g. resumed from a 5M-step checkpoint),
train_log.csv will have a large jump in the timestep column between
sessions. We detect that jump and break the line there (insert NaN) instead
of drawing a straight connector across it, which would otherwise look like
a real (and misleading) cost trajectory.

Usage:
    python plot_live.py ppo                    # log-log (default), live
    python plot_live.py all                    # all techniques, log-log
    python plot_live.py ppo --scale linear      # plain linear-linear
    python plot_live.py ppo --scale logx        # log timestep, linear cost
    python plot_live.py ppo --scale logy        # linear timestep, log cost
    python plot_live.py ppo --save              # save PNG instead of live
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

# If the gap between two consecutive rows' timestep values is more than
# this multiple of the median gap, treat it as a session boundary (e.g.
# jumping from timestep ~550K of an old run to timestep ~5M of a
# resumed/migrated run) and break the plotted line there.
GAP_MULTIPLIER = 20

VALID_SCALES = ["loglog", "linear", "logx", "logy"]


# ---------------------------------------------------------------------------
# K/M tick formatting
# ---------------------------------------------------------------------------

def _human_format(value, _pos=None):
    """Format a number as e.g. 4000 -> '4K', 5000000 -> '5M'."""
    value = float(value)
    sign = "-" if value < 0 else ""
    value = abs(value)
    if value >= 1_000_000_000:
        s = f"{value / 1_000_000_000:g}B"
    elif value >= 1_000_000:
        s = f"{value / 1_000_000:g}M"
    elif value >= 1_000:
        s = f"{value / 1_000:g}K"
    else:
        s = f"{value:g}"
    return sign + s


def _apply_human_formatter(axis):
    from matplotlib.ticker import FuncFormatter
    axis.set_major_formatter(FuncFormatter(_human_format))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_train_log(tech_name):
    log_path = MODELS_DIR / tech_name / "train_log.csv"
    if not log_path.exists() or log_path.stat().st_size < 10:
        return None
    try:
        df = pd.read_csv(log_path)
        if len(df) == 0:
            return None
        return df
    except Exception:
        return None


def prepare_plot_series(df):
    """Return (x, y) with NaN inserted at session-boundary gaps.

    Uses 'timestep' as the x-axis rather than 'episode', since timestep
    is the quantity we checkpoint against and is monotonic within a
    session even when episode numbering resets or gets estimated (as it
    is for migrated pre-trained models).
    """
    df = df.sort_values("timestep").reset_index(drop=True)

    smoothed = df["episode_cost"].rolling(ROLLING_WINDOW, min_periods=1).mean()
    x = df["timestep"].to_numpy(dtype=float)
    y = smoothed.to_numpy(dtype=float)

    if len(x) < 2:
        return x, y

    gaps = np.diff(x)
    median_gap = np.median(gaps[gaps > 0]) if np.any(gaps > 0) else 1.0
    threshold = max(median_gap * GAP_MULTIPLIER, 1.0)

    break_points = np.where(gaps > threshold)[0]
    if len(break_points) == 0:
        return x, y

    # Insert a NaN row right after each break point so matplotlib lifts
    # the pen instead of drawing a straight connector across the gap.
    x_out, y_out = [], []
    prev = 0
    for bp in break_points:
        x_out.extend(x[prev:bp + 1])
        y_out.extend(y[prev:bp + 1])
        x_out.append(np.nan)
        y_out.append(np.nan)
        prev = bp + 1
    x_out.extend(x[prev:])
    y_out.extend(y[prev:])

    return np.array(x_out), np.array(y_out)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_one_axis(ax, tech_name, df, scale="loglog"):
    ax.clear()

    use_logx = scale in ("loglog", "logx")
    use_logy = scale in ("loglog", "logy")

    if df is not None and len(df) > 0:
        x, y = prepare_plot_series(df)

        # log scale can't plot x=0 or y<=0; drop those points (timestep 0
        # doesn't occur in practice since the first logged point is after
        # one rollout, and cost is always > 0 by construction)
        if use_logx:
            x = np.where(x <= 0, np.nan, x)
        if use_logy:
            y = np.where(y <= 0, np.nan, y)

        ax.plot(x, y, linewidth=0.8)
        if use_logx:
            ax.set_xscale("log")
        if use_logy:
            ax.set_yscale("log")

        ax.set_xlabel("Timestep")
        ax.set_ylabel("Episode Cost")
        _apply_human_formatter(ax.xaxis)
        _apply_human_formatter(ax.yaxis)

        finite_y = y[~np.isnan(y)]
        if len(finite_y) > 0:
            current = finite_y[-1]
            best = finite_y.min()
            ax.set_title(f"{tech_name.upper()}\n"
                         f"current: {_human_format(current)}  "
                         f"best: {_human_format(best)}")
        else:
            ax.set_title(tech_name.upper())

        n_segments = np.sum(np.isnan(y)) + 1
        if n_segments > 1:
            ax.text(0.02, 0.02, f"{n_segments} training sessions "
                                 f"(gaps not connected)",
                    transform=ax.transAxes, fontsize=7, color="gray",
                    va="bottom")
    else:
        ax.set_title(f"{tech_name.upper()}\n(no data)")

    ax.grid(True, which="both" if (use_logx or use_logy) else "major",
            alpha=0.3)


def plot_techniques(techniques, save=False, scale="loglog"):
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
            plot_one_axis(axes[i], tech, df, scale=scale)
        plt.suptitle(f"Training Progress ({scale})", y=1.02)
        plt.tight_layout()
        out = PLOTS_DIR / f"training_progress_{scale}.png"
        plt.savefig(str(out), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {out}")
        return

    # --- Live mode ---
    plt.ion()
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    axes = axes[0]

    print(f"Live plotter running (scale={scale}). "
          f"Close the window or Ctrl+C to stop.")
    try:
        while True:
            for i, tech in enumerate(techniques):
                df = load_train_log(tech)
                plot_one_axis(axes[i], tech, df, scale=scale)

            plt.suptitle(
                f"Training Progress ({scale})  (refreshing every 30s)",
                y=1.02)
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
    parser.add_argument("--scale", choices=VALID_SCALES, default="loglog",
                        help="Axis scale (default: loglog). "
                             "Use 'linear' for plain axes, 'logx'/'logy' "
                             "to log only one axis.")
    args = parser.parse_args()

    techniques = all_techniques if args.technique == "all" else [args.technique]
    plot_techniques(techniques, save=args.save, scale=args.scale)


if __name__ == "__main__":
    main()
