"""Wandb integration helper — graceful no-op if wandb is unavailable.

W&B is an optional visualization mirror only. Local training logs remain the
source of truth. Normal W&B writes are time-throttled to at most one emission
every five seconds; the latest pending metrics are merged and force-flushed at
finish. Any W&B error disables further W&B logging without affecting training.
"""

import platform
import threading
import time

_wandb = None
_enabled = False
_run = None

_LOG_INTERVAL_S = 5.0
_last_emit_monotonic = None
_pending_data = {}
_pending_step = None
_lock = threading.Lock()
_error_reported = False


def _reset_throttle_state():
    global _last_emit_monotonic, _pending_data, _pending_step, _error_reported
    _last_emit_monotonic = None
    _pending_data = {}
    _pending_step = None
    _error_reported = False


def init(technique_name, config, tech_config):
    """Start a wandb run. Call at the beginning of training."""
    global _wandb, _enabled, _run

    _reset_throttle_state()

    # Check if wandb is enabled in config.
    wandb_config = config.get("wandb", {})
    if not wandb_config.get("enabled", True):
        _enabled = False
        _run = None
        return

    try:
        import wandb
        _wandb = wandb
    except ImportError:
        print("  [wandb] not installed - skipping. Install with: pip install wandb")
        _enabled = False
        _run = None
        return

    project = wandb_config.get("project", "rl-inventory")
    hostname = platform.node()
    run_name = f"{technique_name}-{hostname}"

    # Flatten tech_config for wandb config (it doesn't like nested dicts well).
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
        print(f"  [wandb] metric mirror interval: {_LOG_INTERVAL_S:.0f}s")
    except Exception as e:
        print(f"  [wandb] init failed: {e} - continuing without wandb")
        _enabled = False
        _run = None


def _disable_after_error(exc):
    global _enabled, _error_reported
    if not _error_reported:
        print(f"  [wandb] logging failed: {exc} - disabling W&B for this run")
        _error_reported = True
    _enabled = False


def _emit_pending_locked(force=False):
    """Emit buffered metrics if the throttle interval has elapsed.

    Caller must hold _lock. ``force=True`` is used only during finish so the
    final pending state is not lost.
    """
    global _last_emit_monotonic, _pending_data, _pending_step

    if not _enabled or _wandb is None or not _pending_data:
        return

    now = time.monotonic()
    if (
        not force
        and _last_emit_monotonic is not None
        and now - _last_emit_monotonic < _LOG_INTERVAL_S
    ):
        return

    data = dict(_pending_data)
    step = _pending_step
    _pending_data = {}
    _pending_step = None

    try:
        if step is not None:
            _wandb.log(data, step=step)
        else:
            _wandb.log(data)
        _last_emit_monotonic = now
    except Exception as exc:
        _disable_after_error(exc)


def log(data, step=None):
    """Queue metrics and mirror them to W&B at most once every five seconds.

    Calls may happen every episode. Values received during the throttle window
    are merged so the next W&B point contains the latest value for each metric.
    The local CSV/JSON logging path is untouched and remains full fidelity.
    """
    global _pending_step

    if not _enabled or _wandb is None:
        return
    if not isinstance(data, dict) or not data:
        return

    with _lock:
        _pending_data.update(data)
        if step is not None:
            try:
                numeric_step = int(step)
            except Exception:
                numeric_step = step
            if _pending_step is None:
                _pending_step = numeric_step
            else:
                try:
                    _pending_step = max(_pending_step, numeric_step)
                except Exception:
                    _pending_step = numeric_step
        _emit_pending_locked(force=False)


def log_episode(episode, timestep, episode_cost, rolling_avg=None,
                best_cost=None):
    """Convenience: queue standard per-episode metrics."""
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
    """Convenience: queue checkpoint-level summary."""
    data = {
        "checkpoint_steps": steps,
        "checkpoint_episodes": episodes,
        "converged": int(converged),
    }
    if best_cost is not None and best_cost < float("inf"):
        data["best_cost_at_checkpoint"] = best_cost
    log(data, step=steps)


def finish():
    """Flush the latest pending metrics and end the W&B run."""
    global _enabled, _run, _wandb

    # Force-flush only if logging is still healthy. This may occur less than
    # five seconds after the preceding emission, intentionally, so the final
    # state is never lost.
    if _enabled and _wandb is not None:
        with _lock:
            _emit_pending_locked(force=True)

    # Even if a previous log call failed and disabled logging, still make a
    # best-effort finish on an initialized run.
    if _wandb is not None and _run is not None:
        try:
            _wandb.finish()
        except Exception:
            pass

    _enabled = False
    _run = None
    _reset_throttle_state()
