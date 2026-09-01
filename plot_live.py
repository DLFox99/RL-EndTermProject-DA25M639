#!/usr/bin/env python3
"""Live RL training and diagnostic dashboards.

Single-technique mode shows the Phase-3 algorithm-specific diagnostic panel in
an intentionally balanced row/column grid. ``all`` mode remains a compact
performance overview (one cost/evaluation panel per technique), also arranged
as a balanced grid.

Examples:
    python plot_live.py nn_sarsa
    python plot_live.py ppo --scale linear
    python plot_live.py all --save
"""

import argparse
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import yaml

from diagnostics import PANEL_SPECS, subplot_grid, x_column_for

PROJECT_ROOT = Path(__file__).parent
MODELS_DIR = PROJECT_ROOT / "models"
PLOTS_DIR = PROJECT_ROOT / "plots"
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

ROLLING_WINDOW = 100
GAP_MULTIPLIER = 20
VALID_SCALES = ["loglog", "linear", "logx", "logy"]
REFRESH_SECONDS = 30


def _human_format(value, _pos=None):
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


def _load_csv(path):
    if not path.exists() or path.stat().st_size < 10:
        return None
    try:
        df = pd.read_csv(path)
        return df if len(df) else None
    except Exception:
        return None


def load_train_log(technique):
    return _load_csv(MODELS_DIR / technique / "train_log.csv")


def load_eval_log(technique):
    return _load_csv(MODELS_DIR / technique / "eval_metrics.csv")


def load_diagnostics(technique):
    return _load_csv(MODELS_DIR / technique / "diagnostics.csv")


def _series_with_session_breaks(df, x_col, y_col, rolling=None):
    if df is None or x_col not in df.columns or y_col not in df.columns:
        return np.array([]), np.array([])
    data = pd.DataFrame({
        "x": pd.to_numeric(df[x_col], errors="coerce"),
        "y": pd.to_numeric(df[y_col], errors="coerce"),
    }).dropna().sort_values("x").reset_index(drop=True)
    if not len(data):
        return np.array([]), np.array([])
    if rolling:
        data["y"] = data["y"].rolling(rolling, min_periods=1).mean()
    x = data["x"].to_numpy(dtype=float)
    y = data["y"].to_numpy(dtype=float)
    if len(x) < 2:
        return x, y

    gaps = np.diff(x)
    positives = gaps[gaps > 0]
    median_gap = np.median(positives) if len(positives) else 1.0
    threshold = max(float(median_gap) * GAP_MULTIPLIER, 1.0)
    breaks = np.where(gaps > threshold)[0]
    if not len(breaks):
        return x, y

    xo, yo, start = [], [], 0
    for bp in breaks:
        xo.extend(x[start:bp + 1]); yo.extend(y[start:bp + 1])
        xo.append(np.nan); yo.append(np.nan)
        start = bp + 1
    xo.extend(x[start:]); yo.extend(y[start:])
    return np.asarray(xo), np.asarray(yo)


def _apply_scale(ax, scale, *, allow_log_y=True):
    if scale in ("loglog", "logx"):
        ax.set_xscale("log")
    if allow_log_y and scale in ("loglog", "logy"):
        ax.set_yscale("log")


def _cost_axis(ax, technique, train_df, eval_df, scale):
    ax.clear()
    x_col = x_column_for(technique)
    x, y = _series_with_session_breaks(
        train_df, x_col, "episode_cost", rolling=ROLLING_WINDOW)

    use_logx = scale in ("loglog", "logx")
    use_logy = scale in ("loglog", "logy")
    if use_logx and len(x):
        x = np.where(x <= 0, np.nan, x)
    if use_logy and len(y):
        y = np.where(y <= 0, np.nan, y)

    if len(x):
        ax.plot(x, y, linewidth=0.9, label="training rolling-100")

    eval_best = None
    if eval_df is not None and x_col in eval_df.columns and "mean_cost" in eval_df.columns:
        ex = pd.to_numeric(eval_df[x_col], errors="coerce").to_numpy(dtype=float)
        ey = pd.to_numeric(eval_df["mean_cost"], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(ex) & np.isfinite(ey)
        if use_logx:
            valid &= ex > 0
        if use_logy:
            valid &= ey > 0
        ex, ey = ex[valid], ey[valid]
        order = np.argsort(ex)
        ex, ey = ex[order], ey[order]
        if len(ex):
            ax.plot(ex, ey, marker="o", markersize=3, linewidth=0.8,
                    linestyle="--", label="deterministic eval")
            eval_best = float(np.min(ey))

        # Phase 4: mark the first advisory deterministic-evaluation plateau.
        if "plateau_detected" in eval_df.columns:
            detected = pd.to_numeric(
                eval_df["plateau_detected"], errors="coerce").fillna(0) > 0
            if detected.any():
                first_idx = detected[detected].index[0]
                try:
                    plateau_x = float(eval_df.loc[first_idx, x_col])
                    if np.isfinite(plateau_x) and (not use_logx or plateau_x > 0):
                        ax.axvline(plateau_x, linestyle=":", linewidth=0.9,
                                   label="eval plateau")
                except Exception:
                    pass

    _apply_scale(ax, scale, allow_log_y=True)
    ax.set_xlabel("Episode" if x_col == "episode" else "Timestep")
    ax.set_ylabel("Episode Cost")
    _apply_human_formatter(ax.xaxis)
    _apply_human_formatter(ax.yaxis)

    finite_y = y[np.isfinite(y)] if len(y) else np.array([])
    if len(finite_y):
        title = (f"Cost + Evaluation\ncurrent={_human_format(finite_y[-1])}  "
                 f"train-best={_human_format(np.min(finite_y))}")
        if eval_best is not None:
            title += f"  eval-best={_human_format(eval_best)}"
    else:
        title = "Cost + Evaluation\n(no training data)"
    ax.set_title(title, fontsize=9)
    if len(ax.lines) > 1:
        ax.legend(fontsize=7)
    ax.grid(True, which="both" if (use_logx or use_logy) else "major", alpha=0.3)


def _diagnostic_axis(ax, title, metric_names, diag_df, technique, scale):
    ax.clear()
    x_col = x_column_for(technique)
    plotted = 0
    if diag_df is not None and x_col in diag_df.columns:
        x_all = pd.to_numeric(diag_df[x_col], errors="coerce")
        for metric in metric_names:
            if metric not in diag_df.columns:
                continue
            y_all = pd.to_numeric(diag_df[metric], errors="coerce")
            valid = x_all.notna() & y_all.notna()
            if not valid.any():
                continue
            x = x_all[valid].to_numpy(dtype=float)
            y = y_all[valid].to_numpy(dtype=float)
            order = np.argsort(x)
            x, y = x[order], y[order]
            if scale in ("loglog", "logx"):
                keep = x > 0
                x, y = x[keep], y[keep]
            if len(x):
                ax.plot(x, y, linewidth=0.9, label=metric)
                plotted += 1

    # Diagnostic quantities can be negative (policy loss, advantage, entropy
    # loss), so never apply log-y automatically. X follows the requested scale.
    _apply_scale(ax, scale, allow_log_y=False)
    ax.set_xlabel("Episode" if x_col == "episode" else "Timestep")
    ax.set_title(title, fontsize=9)
    _apply_human_formatter(ax.xaxis)
    if plotted > 1:
        ax.legend(fontsize=7)
    elif plotted == 0:
        ax.text(0.5, 0.5, "no diagnostic data", ha="center", va="center",
                transform=ax.transAxes, fontsize=8)
    ax.grid(True, which="both" if scale in ("loglog", "logx") else "major", alpha=0.3)


def _draw_dashboard(axes, technique, scale):
    train_df = load_train_log(technique)
    eval_df = load_eval_log(technique)
    diag_df = load_diagnostics(technique)
    specs = PANEL_SPECS.get(technique, [("Cost + Evaluation", [])])

    flat = list(np.asarray(axes, dtype=object).reshape(-1))
    for i, (title, metrics) in enumerate(specs):
        if i == 0:
            _cost_axis(flat[i], technique, train_df, eval_df, scale)
        else:
            _diagnostic_axis(flat[i], title, metrics, diag_df, technique, scale)
    for ax in flat[len(specs):]:
        ax.set_visible(False)


def _draw_overview(axes, techniques, scale):
    flat = list(np.asarray(axes, dtype=object).reshape(-1))
    for i, technique in enumerate(techniques):
        _cost_axis(flat[i], technique, load_train_log(technique),
                   load_eval_log(technique), scale)
        flat[i].set_title(technique.upper(), fontsize=9)
    for ax in flat[len(techniques):]:
        ax.set_visible(False)


def _figure_shape(techniques):
    if len(techniques) == 1:
        n = len(PANEL_SPECS.get(techniques[0], [("Cost + Evaluation", [])]))
    else:
        n = len(techniques)
    return subplot_grid(n)


def plot_techniques(techniques, save=False, scale="loglog"):
    if save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PLOTS_DIR.mkdir(exist_ok=True)
    rows, cols = _figure_shape(techniques)
    fig, axes = plt.subplots(
        rows, cols,
        figsize=(4.7 * cols, 3.5 * rows),
        squeeze=False,
    )

    def redraw():
        if len(techniques) == 1:
            _draw_dashboard(axes, techniques[0], scale)
            fig.suptitle(
                f"{techniques[0].upper()} Diagnostic Dashboard ({scale})",
                y=0.995)
        else:
            _draw_overview(axes, techniques, scale)
            fig.suptitle(f"Training Performance Overview ({scale})", y=0.995)
        fig.tight_layout(rect=(0, 0, 1, 0.97))

    redraw()

    if save:
        if len(techniques) == 1:
            out = PLOTS_DIR / f"diagnostics_{techniques[0]}_{scale}.png"
        else:
            out = PLOTS_DIR / f"training_progress_{scale}.png"
        fig.savefig(str(out), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out}")
        return

    plt.ion()
    print(f"Live plotter running (scale={scale}, refresh={REFRESH_SECONDS}s). "
          "Close the window or Ctrl+C to stop.")
    try:
        while plt.fignum_exists(fig.number):
            redraw()
            plt.pause(REFRESH_SECONDS)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        plt.close("all")


def main():
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    all_techniques = list(config["techniques"].keys())

    parser = argparse.ArgumentParser(description="Plot RL training diagnostics")
    parser.add_argument("technique", choices=all_techniques + ["all"])
    parser.add_argument("--save", action="store_true",
                        help="Save PNG instead of live display")
    parser.add_argument("--scale", choices=VALID_SCALES, default="loglog",
                        help="Cost/x scale. Diagnostic y-axes remain linear because they may be signed.")
    args = parser.parse_args()

    techniques = all_techniques if args.technique == "all" else [args.technique]
    plot_techniques(techniques, save=args.save, scale=args.scale)


if __name__ == "__main__":
    main()
