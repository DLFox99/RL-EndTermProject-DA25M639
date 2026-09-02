#!/usr/bin/env python3
import csv
import json
import tempfile
from pathlib import Path

import yaml

import assemble
import candidate_pipeline
from candidate_pipeline import collect_candidates, collect_direct_candidates


def test_assemble_auto_prefers_best_eval():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        models = root / "models"
        subs = root / "subs"
        model_dir = models / "nn_sarsa"
        model_dir.mkdir(parents=True)
        (model_dir / "best_eval_model.pt").write_bytes(b"eval")
        (model_dir / "best_model.pt").write_bytes(b"train")
        (model_dir / "final_model.pt").write_bytes(b"final")
        tc = {"category": "nn_custom", "portal_name": "NN SARSA", "hidden": 128}
        ok = assemble.assemble_technique(
            "nn_sarsa", tc, models_root=models, submissions_root=subs,
            checkpoint="auto",
        )
        assert ok
        assert (subs / "nn_sarsa" / "nn_sarsa_model.pt").read_bytes() == b"eval"


def test_assemble_explicit_best_train():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        models = root / "models"
        subs = root / "subs"
        model_dir = models / "nn_sarsa"
        model_dir.mkdir(parents=True)
        (model_dir / "best_eval_model.pt").write_bytes(b"eval")
        (model_dir / "best_model.pt").write_bytes(b"train")
        tc = {"category": "nn_custom", "portal_name": "NN SARSA", "hidden": 128}
        ok = assemble.assemble_technique(
            "nn_sarsa", tc, models_root=models, submissions_root=subs,
            checkpoint="best_train",
        )
        assert ok
        assert (subs / "nn_sarsa" / "nn_sarsa_model.pt").read_bytes() == b"train"


def test_collect_candidates_combines_and_ranks_shards():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        tc = {"category": "onpolicy"}
        sweeps = []
        rows_by_sweep = [
            [("trial_001", 300.0), ("trial_003", 100.0)],
            [("trial_002", 200.0)],
        ]
        for i, rows in enumerate(rows_by_sweep):
            sweep = root / f"sweep_{i}"
            sweep.mkdir()
            (sweep / "resolved_sweep.yaml").write_text(
                yaml.safe_dump({"technique": "ppo"})
            )
            with (sweep / "sweep_summary.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "trial_id", "returncode", "best_eval_cost",
                    "best_training_rolling_cost", "steps_completed",
                    "episodes_completed", "overrides",
                ])
                writer.writeheader()
                for trial_id, cost in rows:
                    writer.writerow({
                        "trial_id": trial_id,
                        "returncode": 0,
                        "best_eval_cost": cost,
                        "best_training_rolling_cost": cost + 10,
                        "steps_completed": 1000,
                        "episodes_completed": 20,
                        "overrides": json.dumps({"x": cost}),
                    })
                    model_dir = sweep / trial_id / "models" / "ppo"
                    model_dir.mkdir(parents=True)
                    (model_dir / "best_eval_model.zip").write_bytes(b"model")
            sweeps.append(sweep)

        got = collect_candidates(sweeps, "ppo", tc)
        assert [r["trial_id"] for r in got] == ["trial_003", "trial_002", "trial_001"]
        assert [r["selection_rank"] for r in got] == [1, 2, 3]


def test_independent_evaluation_creates_output_directory():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        output_dir = root / "batch"
        output_dir.mkdir()

        candidate = {
            "selection_rank": 1,
            "trial_id": "trial_007",
            "model_dir": str(root / "models" / "ppo"),
        }

        original_build = candidate_pipeline.build_policy_fn
        original_run = candidate_pipeline.run_policy_evaluation

        try:
            candidate_pipeline.build_policy_fn = (
                lambda *args, **kwargs: lambda obs: [0, 0, 0]
            )
            candidate_pipeline.run_policy_evaluation = (
                lambda *args, **kwargs: {
                    "records": [
                        {"total_cost": 100.0, "seed": 1},
                        {"total_cost": 120.0, "seed": 2},
                    ]
                }
            )

            result = candidate_pipeline._evaluate_candidate(
                candidate,
                "ppo",
                {},
                {
                    "evaluation": {
                        "seeds": [1, 2],
                        "scenario_modes": ["normal"],
                    }
                },
                {},
                output_dir,
            )

            eval_file = Path(result["evaluation_file"])
            assert eval_file.exists()
            assert eval_file.parent == output_dir / "evaluations"
            assert result["independent_mean_cost"] == 110.0
            assert result["independent_n_episodes"] == 2
        finally:
            candidate_pipeline.build_policy_fn = original_build
            candidate_pipeline.run_policy_evaluation = original_run



def test_collect_direct_candidate():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        model_dir = root / "runA" / "models" / "nn_sarsa"
        model_dir.mkdir(parents=True)
        (model_dir / "best_eval_model.pt").write_bytes(b"model")
        (model_dir / "evaluation_metadata.json").write_text(
            json.dumps({"best_mean_cost": 123.5})
        )
        (model_dir / "training_metadata.json").write_text(
            json.dumps({"best_rolling_cost": 130.0, "steps_completed": 1000})
        )
        (model_dir / "hyperparams_used.yaml").write_text("epsilon_end: 0.02\n")
        got = collect_direct_candidates(
            [model_dir], "nn_sarsa", {"category": "nn_custom"}
        )
        assert len(got) == 1
        assert got[0]["selection_best_eval_cost"] == 123.5
        assert got[0]["models_root"] == str(model_dir.parent)
        assert got[0]["overrides"]["epsilon_end"] == 0.02


if __name__ == "__main__":
    test_assemble_auto_prefers_best_eval()
    test_assemble_explicit_best_train()
    test_collect_candidates_combines_and_ranks_shards()
    test_independent_evaluation_creates_output_directory()
    test_collect_direct_candidate()
    print("PASS Phase 6 candidate pipeline tests")
