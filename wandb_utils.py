"""Wandb integration helper — graceful no-op if wandb is not installed.

All training code calls these functions. If wandb is not installed or
disabled in config.yaml, they silently do nothing — training works
identically either way.
"""

import platform
from pathlib import Path

_wandb = None
_enabled = False
_run = None


def init(technique_name, config, tech_config):
    """Start a wandb run. Call at the beginning of training."""
    global _wandb, _enabled, _run

    # Check if wandb is enabled in config
    wandb_config = config.get("wandb", {})
    if not wandb_config.get("enabled", True):
        _enabled = False
        return

    try:
        import wandb
        _wandb = wandb
    except ImportError:
        print("  [wandb] not installed — skipping. Install with: pip install wandb")
        _enabled = False
        return

    project = wandb_config.get("project", "rl-inventory")
    hostname = platform.node()
    run_name = f"{technique_name}-{hostname}"

    # Flatten tech_config for wandb config (it doesn't like nested dicts well)
    flat_config = {
        "technique": technique_name,
        "machine": hostname,
        **{k: v for k, v in tech_config.items()
           if not isinstance(v, (dict, list)) or k in ("net_arch",)},
    }

    try:
        _run = wandb.init(
            project=project,
            name=run_name,
            group=technique_name,
            tags=[technique_name, hostname],
            config=flat_config,
            reinit=True,
        )
        _enabled = True
        print(f"  [wandb] logging to project '{project}' as '{run_name}'")
        print(f"  [wandb] dashboard: {getattr(_run, 'url', None)}")
    except Exception as e:
        print(f"  [wandb] init failed: {e} — continuing without wandb")
        _enabled = False


def log(data, step=None):
    """Log a dict of metrics. Call per episode or per checkpoint."""
    if not _enabled or _wandb is None:
        return
    try:
        if step is not None:
            _wandb.log(data, step=step)
        else:
            _wandb.log(data)
    except Exception:
        pass  # never crash training because of a logging failure


def log_episode(episode, timestep, episode_cost, rolling_avg=None,
                best_cost=None):
    """Convenience: log standard per-episode metrics."""
    data = {
        "episode_cost": episode_cost,
        "timestep": timestep,
        "episode": episode,
    }
    if rolling_avg is not None:
        data["rolling_avg_cost"] = rolling_avg
    if best_cost is not None and best_cost < float("inf"):
        data["best_cost"] = best_cost
    log(data, step=timestep)


def log_checkpoint(steps, episodes, best_cost, converged=False):
    """Convenience: log checkpoint-level summary."""
    data = {
        "checkpoint_steps": steps,
        "checkpoint_episodes": episodes,
        "converged": int(converged),
    }
    if best_cost is not None and best_cost < float("inf"):
        data["best_cost_at_checkpoint"] = best_cost
    log(data, step=steps)


def finish():
    """End the wandb run. Call at the end of training."""
    global _enabled, _run
    if not _enabled or _wandb is None:
        return
    try:
        _wandb.finish()
    except Exception:
        pass
    _enabled = False
    _run = None
