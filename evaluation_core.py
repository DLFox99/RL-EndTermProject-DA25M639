#!/usr/bin/env python3
"""Shared deterministic evaluation primitives for training and offline evaluation.

Phase 2 separates *training performance* from *deployment/evaluation performance*.
The functions here deliberately avoid W&B and write no files by themselves.
"""

from __future__ import annotations

import random
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Sequence

import numpy as np



PolicyFn = Callable[[dict], Sequence[int]]


@contextmanager
def preserve_rng_state():
    """Preserve process-global RNG state so evaluation cannot perturb training.

    The inventory environment is seeded explicitly for every evaluation episode,
    but policy adapters or dependencies may still touch Python/NumPy/Torch global
    RNGs. Restoring those states keeps an evaluation hook observational.
    """
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_mod = None
    torch_cpu_state = None
    torch_cuda_states = None
    try:
        # Never import Torch solely for evaluation bookkeeping. If the trainer
        # already uses Torch it will be present and we preserve its RNG state;
        # tabular-only runs stay lightweight.
        torch_mod = sys.modules.get("torch")
        if torch_mod is not None:
            try:
                torch_cpu_state = torch_mod.random.get_rng_state()
                if torch_mod.cuda.is_available():
                    torch_cuda_states = torch_mod.cuda.get_rng_state_all()
            except Exception:
                torch_mod = None

        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        if torch_mod is not None and torch_cpu_state is not None:
            try:
                torch_mod.random.set_rng_state(torch_cpu_state)
                if torch_cuda_states is not None and torch_mod.cuda.is_available():
                    torch_mod.cuda.set_rng_state_all(torch_cuda_states)
            except Exception:
                pass


def evaluation_cases(seeds: Iterable[int], scenario_modes: Iterable[str]):
    """Return deterministic (scenario_mode, seed) cases in stable order."""
    return [(str(mode), int(seed)) for mode in scenario_modes for seed in seeds]


def run_policy_evaluation(
    policy_fn: PolicyFn,
    student_config: Any,
    seeds: Iterable[int],
    scenario_modes: Iterable[str],
    env_factory=None,
) -> Dict[str, Any]:
    """Evaluate a deterministic policy and return raw records + statistics.

    `policy_fn(obs)` must return order quantities [0, 10, ..., 100]. The
    environment itself consumes action indices [0, ..., 10], matching the
    existing offline evaluator and portal policy convention.
    """
    records: List[Dict[str, Any]] = []

    if env_factory is None:
        from industrial_inventory_env import IndustrialInventoryEnv

        def env_factory(mode):
            return IndustrialInventoryEnv(
                student_config,
                scenario_mode=mode,
                domain_randomization=True,
            )

    with preserve_rng_state():
        for mode, seed in evaluation_cases(seeds, scenario_modes):
            env = env_factory(mode)
            try:
                obs, _ = env.reset(seed=seed)
                done = False
                step_info = None
                while not done:
                    quantities = policy_fn(obs)
                    action_indices = np.asarray(
                        [int(q) // 10 for q in quantities], dtype=np.int64
                    )
                    obs, reward, terminated, truncated, step_info = env.step(
                        action_indices
                    )
                    done = bool(terminated or truncated)

                if step_info is None or "costs" not in step_info:
                    raise RuntimeError("evaluation episode ended without costs")
                costs = step_info["costs"]
                records.append(
                    {
                        "scenario_mode": mode,
                        "seed": seed,
                        "total_cost": float(costs["episode_total"]),
                        "holding_cost": float(costs["holding"]),
                        "stockout_cost": float(costs["stockout"]),
                        "ordering_cost": float(costs["ordering"]),
                        "discard_cost": float(costs["discarding"]),
                    }
                )
            finally:
                try:
                    env.close()
                except Exception:
                    pass

    totals = np.asarray([r["total_cost"] for r in records], dtype=np.float64)
    if totals.size == 0:
        raise RuntimeError("evaluation produced no episodes")

    return {
        "n_episodes": int(totals.size),
        "mean_cost": float(np.mean(totals)),
        "median_cost": float(np.median(totals)),
        "std_cost": float(np.std(totals, ddof=0)),
        "min_cost": float(np.min(totals)),
        "max_cost": float(np.max(totals)),
        "records": records,
    }
