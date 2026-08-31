#!/usr/bin/env python3
"""Check training convergence for one or all techniques.

Uses two signals (both must trigger to declare convergence):
  1. Slope:    Linear trend of rolling-avg cost over the last 30% of the
               current training session. If the normalized improvement rate
               is below a threshold, the cost curve is flat.
  2. Patience: How far back (as a fraction of the current session) the
               best rolling-avg cost was last seen. If it hasn't improved
               for a long stretch, training is stale.

Usage:
    python check_convergence.py ppo
    python check_convergence.py all
    python check_convergence.py all --slope-threshold 0.02
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).parent
MODELS_DIR = PROJECT_ROOT / "models"
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

# Same gap-detection constant as plot_live.py
GAP_MULTIPLIER = 20
ROLLING_WINDOW = 100

# Default thresholds
DEFAULT_SLOPE_THRESHOLD = 0.01      # 1% improvement per 1000 episodes
DEFAULT_PATIENCE_FRACTION = 0.20    # best model stale for 20% of session
DEFAULT_MIN_EPISODES = 500          # need at least this many to judge


def _isolate_latest_session(df):
    """Return only the rows from the most recent contiguous training session."""
    if len(df) < 2:
        return df

    df = df.sort_values("timestep").reset_index(drop=True)
    gaps = np.diff(df["timestep"].values)
    median_gap = np.median(gaps[gaps > 0]) if np.any(gaps > 0) else 1.0
    threshold = max(median_gap * GAP_MULTIPLIER, 1.0)

    break_points = np.where(gaps > threshold)[0]
    if len(break_points) == 0:
        return df

    # Everything after the last gap is the latest session
    last_break = break_points[-1] + 1
    return df.iloc[last_break:].reset_index(drop=True)


def check_convergence(
    train_log_path,
    slope_threshold=DEFAULT_SLOPE_THRESHOLD,
    patience_fraction=DEFAULT_PATIENCE_FRACTION,
    min_episodes=DEFAULT_MIN_EPISODES,
    tail_fraction=0.30,
):
    """Analyze convergence from a train_log.csv file.

    Returns
    -------
    dict with keys:
        converged        : bool — True only if BOTH slope and patience trigger
        slope_flat       : bool — slope signal alone
        patience_stale   : bool — patience signal alone
        slope_pct        : float — normalized improvement rate (% per 1000 ep)
        stale_fraction   : float — fraction of session since best was seen
        current_cost     : float — rolling-avg cost at end of session
        best_cost        : float — best rolling-avg cost in session
        session_episodes : int   — number of episodes in latest session
        reason           : str   — human-readable summary
    """
    result = {
        "converged": False,
        "slope_flat": False,
        "patience_stale": False,
        "slope_pct": None,
        "stale_fraction": None,
        "current_cost": None,
        "best_cost": None,
        "session_episodes": 0,
        "reason": "",
    }

    # Load and isolate latest session
    log_path = Path(train_log_path)
    if not log_path.exists() or log_path.stat().st_size < 10:
        result["reason"] = "no training log found"
        return result

    try:
        df = pd.read_csv(log_path)
    except Exception as e:
        result["reason"] = f"failed to read log: {e}"
        return result

    if len(df) == 0:
        result["reason"] = "empty training log"
        return result

    session = _isolate_latest_session(df)
    n_episodes = len(session)
    result["session_episodes"] = n_episodes

    if n_episodes < min_episodes:
        result["reason"] = (f"too few episodes in current session "
                            f"({n_episodes} < {min_episodes})")
        return result

    # Compute rolling average
    costs = session["episode_cost"].values.astype(float)
    timesteps = session["timestep"].values.astype(float)
    rolling = pd.Series(costs).rolling(ROLLING_WINDOW, min_periods=1).mean().values

    result["current_cost"] = float(rolling[-1])
    result["best_cost"] = float(np.min(rolling))

    # --- Signal 1: Slope over the tail portion ---
    tail_start = int(n_episodes * (1 - tail_fraction))
    tail_rolling = rolling[tail_start:]
    tail_timesteps = timesteps[tail_start:]

    if len(tail_rolling) < 50:
        result["reason"] = "tail too short for slope calculation"
        return result

    # Normalize timesteps to [0, 1] for numerical stability
    t_norm = (tail_timesteps - tail_timesteps[0])
    t_range = t_norm[-1] - t_norm[0]
    if t_range == 0:
        result["reason"] = "zero timestep range in tail"
        return result
    t_norm = t_norm / t_range

    # Linear regression: cost = a * t_norm + b
    coeffs = np.polyfit(t_norm, tail_rolling, 1)
    slope_raw = coeffs[0]  # change in cost over normalized [0, 1]

    # Express as percentage improvement per 1000 episodes
    mean_cost = np.mean(tail_rolling)
    if mean_cost == 0:
        result["reason"] = "zero mean cost (impossible?)"
        return result

    tail_episodes = len(tail_rolling)
    # slope_raw is total change over the tail window
    # normalize to "per 1000 episodes" and as % of mean cost
    slope_per_1000 = (slope_raw / tail_episodes) * 1000
    slope_pct = -(slope_per_1000 / mean_cost) * 100  # negative slope = improvement

    result["slope_pct"] = float(slope_pct)
    result["slope_flat"] = abs(slope_pct) < (slope_threshold * 100)

    # --- Signal 2: Patience (staleness of best model) ---
    best_idx = int(np.argmin(rolling))
    episodes_since_best = n_episodes - best_idx
    stale_fraction = episodes_since_best / n_episodes

    result["stale_fraction"] = float(stale_fraction)
    result["patience_stale"] = stale_fraction > patience_fraction

    # --- Combined verdict ---
    result["converged"] = result["slope_flat"] and result["patience_stale"]

    # Build reason string
    slope_str = f"slope={slope_pct:+.2f}%/1K ep"
    stale_str = f"best seen {stale_fraction:.0%} of session ago"

    if result["converged"]:
        result["reason"] = f"CONVERGED ({slope_str}, {stale_str})"
    elif result["slope_flat"]:
        result["reason"] = (f"slope flat but best still recent "
                            f"({slope_str}, {stale_str})")
    elif result["patience_stale"]:
        result["reason"] = (f"best is stale but slope still moving "
                            f"({slope_str}, {stale_str})")
    else:
        result["reason"] = f"IMPROVING ({slope_str}, {stale_str})"

    return result


def check_technique(tech_name):
    """Check convergence for one technique, return result dict."""
    log_path = MODELS_DIR / tech_name / "train_log.csv"
    return check_convergence(log_path)


def main():
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    all_techniques = list(config["techniques"].keys())

    parser = argparse.ArgumentParser(description="Check training convergence")
    parser.add_argument("technique", choices=all_techniques + ["all"])
    parser.add_argument("--slope-threshold", type=float,
                        default=DEFAULT_SLOPE_THRESHOLD,
                        help=f"Slope threshold (default: {DEFAULT_SLOPE_THRESHOLD})")
    parser.add_argument("--patience-fraction", type=float,
                        default=DEFAULT_PATIENCE_FRACTION,
                        help=f"Patience fraction (default: {DEFAULT_PATIENCE_FRACTION})")
    parser.add_argument("--min-episodes", type=int,
                        default=DEFAULT_MIN_EPISODES,
                        help=f"Minimum episodes to judge (default: {DEFAULT_MIN_EPISODES})")
    args = parser.parse_args()

    techniques = all_techniques if args.technique == "all" else [args.technique]

    print(f"\n{'Technique':<22s} {'Status':<12s} {'Current':>10s} {'Best':>10s} "
          f"{'Slope':>12s} {'Stale':>8s}  Details")
    print("-" * 100)

    for tech in techniques:
        log_path = MODELS_DIR / tech / "train_log.csv"
        r = check_convergence(
            log_path,
            slope_threshold=args.slope_threshold,
            patience_fraction=args.patience_fraction,
            min_episodes=args.min_episodes,
        )

        if r["session_episodes"] == 0:
            print(f"{tech:<22s} {'NO DATA':<12s} {'—':>10s} {'—':>10s} "
                  f"{'—':>12s} {'—':>8s}  {r['reason']}")
            continue

        status = "CONVERGED" if r["converged"] else "IMPROVING"
        current = f"{r['current_cost']:,.0f}" if r["current_cost"] else "—"
        best = f"{r['best_cost']:,.0f}" if r["best_cost"] else "—"
        slope = f"{r['slope_pct']:+.2f}%/1K" if r["slope_pct"] is not None else "—"
        stale = f"{r['stale_fraction']:.0%}" if r["stale_fraction"] is not None else "—"

        print(f"{tech:<22s} {status:<12s} {current:>10s} {best:>10s} "
              f"{slope:>12s} {stale:>8s}  {r['reason']}")

    print()


if __name__ == "__main__":
    main()
