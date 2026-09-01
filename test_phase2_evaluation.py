#!/usr/bin/env python3
"""Synthetic Phase-2 tests; no Gym/SB3/Torch required."""

import csv
import json
import random
import tempfile
from pathlib import Path

import numpy as np

import periodic_evaluation as pe
from evaluation_core import run_policy_evaluation


class FakeEnv:
    def __init__(self, mode):
        self.mode = mode
        self.seed = None

    def reset(self, seed=None):
        self.seed = int(seed)
        return {"x": 0}, {}

    def step(self, action):
        # One-step deterministic episode. Policy actions are irrelevant.
        offset = {"a": 0, "b": 100}[self.mode]
        total = self.seed + offset
        info = {
            "costs": {
                "episode_total": total,
                "holding": total * 0.1,
                "stockout": total * 0.2,
                "ordering": total * 0.3,
                "discarding": total * 0.4,
            }
        }
        return {"x": 1}, 0.0, True, False, info

    def close(self):
        pass


def test_core_and_rng():
    random.seed(123)
    np.random.seed(456)
    py_before = random.getstate()
    np_before = np.random.get_state()

    def noisy_policy(obs):
        # Deliberately consume global RNG; evaluation_core must restore it.
        random.random()
        np.random.random()
        return [0, 0, 0]

    result = run_policy_evaluation(
        noisy_policy,
        student_config=None,
        seeds=[1, 2],
        scenario_modes=["a", "b"],
        env_factory=lambda mode: FakeEnv(mode),
    )
    assert result["n_episodes"] == 4
    assert result["mean_cost"] == 51.5, result
    assert random.getstate() == py_before
    now = np.random.get_state()
    assert now[0] == np_before[0]
    assert np.array_equal(now[1], np_before[1])
    assert now[2:] == np_before[2:]


def test_periodic_selection():
    with tempfile.TemporaryDirectory() as tmp:
        model_dir = Path(tmp) / "models" / "nn_sarsa"
        model_dir.mkdir(parents=True)

        # Replace the real environment evaluator with deterministic synthetic
        # scores so this test remains dependency-free.
        scores = iter([100.0, 90.0, 95.0])
        original = pe.run_policy_evaluation

        def fake_eval(policy_fn, student_config, seeds, scenario_modes):
            mean = next(scores)
            return {
                "n_episodes": len(seeds) * len(scenario_modes),
                "mean_cost": mean,
                "median_cost": mean,
                "std_cost": 1.0,
                "min_cost": mean - 1,
                "max_cost": mean + 1,
                "records": [{"total_cost": mean}],
            }

        pe.run_policy_evaluation = fake_eval
        try:
            cfg = {
                "evaluation": {
                    "periodic": {
                        "enabled": True,
                        "seeds": [200],
                        "scenario_modes": ["random"],
                        "default_interval_episodes": 10,
                    }
                }
            }
            evaluator = pe.PeriodicEvaluator(
                technique="nn_sarsa",
                config=cfg,
                tech_config={},
                student_config=None,
                model_dir=model_dir,
                progress_unit="episodes",
                force=True,
            )

            save_number = {"value": 0}

            def save_best(path):
                save_number["value"] += 1
                path.with_suffix(".pt").write_text(str(save_number["value"]))

            factory = lambda: (lambda obs: [0, 0, 0])
            assert evaluator.maybe_evaluate(
                progress=9, episode=9, policy_factory=factory,
                save_best_fn=save_best) is None
            evaluator.maybe_evaluate(
                progress=10, episode=10, policy_factory=factory,
                save_best_fn=save_best)
            evaluator.maybe_evaluate(
                progress=20, episode=20, policy_factory=factory,
                save_best_fn=save_best)
            evaluator.evaluate_final(
                progress=25, episode=25, policy_factory=factory,
                save_best_fn=save_best)

            with (model_dir / "eval_metrics.csv").open(newline="") as f:
                rows = list(csv.DictReader(f))
            assert len(rows) == 3
            assert float(rows[0]["mean_cost"]) == 100.0
            assert float(rows[1]["mean_cost"]) == 90.0
            assert float(rows[2]["mean_cost"]) == 95.0
            meta = json.loads((model_dir / "evaluation_metadata.json").read_text())
            assert meta["best_mean_cost"] == 90.0
            # Saved at 100 and 90, not 95.
            assert (model_dir / "best_eval_model.pt").read_text() == "2"
        finally:
            pe.run_policy_evaluation = original


if __name__ == "__main__":
    test_core_and_rng()
    test_periodic_selection()
    print("PASS: Phase-2 deterministic evaluation tests")
