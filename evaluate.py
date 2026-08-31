#!/usr/bin/env python3
"""Evaluate trained models locally.

Usage:
    python evaluate.py ppo          # evaluate one, save CSV
    python evaluate.py all          # evaluate all, generate comparison
    python evaluate.py ppo --force  # re-evaluate even if CSV exists
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from env_wrappers import flatten_observation, DiscreteActionWrapper
from industrial_inventory_env import IndustrialInventoryEnv, generate_student_config

PROJECT_ROOT = Path(__file__).parent
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def build_policy_fn(tech_name, tc, model_dir):
    """Build a policy function that maps observation → [q1, q2, q3]."""
    action_type = tc.get("action_type", "multi")
    category = tc.get("category", "onpolicy")

    if category == "tabular":
        from tabular_agent import StateDiscretizer, TabularAgent

        discretizer = StateDiscretizer()
        agent = TabularAgent(discretizer.n_states)

        model_path = model_dir / "best_model.npz"
        if not model_path.exists():
            model_path = model_dir / "final_model.npz"
        agent.load(str(model_path))

        def fn(obs):
            state = discretizer.discretize(obs)
            actions = agent.get_greedy_actions(state)
            return [a * 10 for a in actions]
        return fn

    elif category in ("reinforce", "a3c"):
        import torch
        if category == "a3c":
            from a3c_agent import ActorCritic as NetClass
        else:
            from reinforce_agent import ReinforcePolicy as NetClass

        model_path = model_dir / "best_model.pt"
        if not model_path.exists():
            model_path = model_dir / "final_model.pt"

        net = NetClass(hidden=tc.get("hidden", 128))
        net.load_state_dict(
            torch.load(str(model_path), map_location="cpu", weights_only=True))
        net.eval()

        def fn(obs):
            flat = flatten_observation(obs)
            obs_t = torch.FloatTensor(flat).unsqueeze(0)
            with torch.no_grad():
                logits = net(obs_t) if category == "reinforce" \
                    else net(obs_t)[0]  # A3C returns (logits, value)
            actions = [int(l.argmax(dim=-1).item()) for l in logits]
            return [a * 10 for a in actions]
        return fn

    elif category == "nn_custom":
        import torch
        from nn_agent import QNetworkFactored

        model_path = model_dir / "best_model.pt"
        if not model_path.exists():
            model_path = model_dir / "final_model.pt"

        net = QNetworkFactored(hidden=tc.get("hidden", 128))
        net.load_state_dict(
            torch.load(str(model_path), map_location="cpu", weights_only=True))
        net.eval()

        def fn(obs):
            flat = flatten_observation(obs)
            obs_t = torch.FloatTensor(flat).unsqueeze(0)
            with torch.no_grad():
                q_values = net(obs_t)
            actions = [int(q.argmax(dim=-1).item()) for q in q_values]
            return [a * 10 for a in actions]
        return fn

    else:
        # SB3 model
        algo_name = tc["algo"]
        if algo_name == "DoubleDQN":
            from stable_baselines3 import DQN as loader_cls
        else:
            import stable_baselines3
            loader_cls = getattr(stable_baselines3, algo_name)

        # Prefer best_model, fall back to final
        model_path = model_dir / "best_model"
        if not (model_dir / "best_model.zip").exists():
            model_path = model_dir / "final_model"

        model = loader_cls.load(str(model_path))

        if action_type == "discrete":
            def fn(obs):
                flat = flatten_observation(obs)
                action, _ = model.predict(flat, deterministic=True)
                return DiscreteActionWrapper.decode_discrete(int(action))
        else:
            def fn(obs):
                flat = flatten_observation(obs)
                action, _ = model.predict(flat, deterministic=True)
                return (np.asarray(action, dtype=np.int64) * 10).tolist()
        return fn


def evaluate_technique(tech_name, tc, config, student_config, force=False):
    """Run policy over all seeds × scenario modes, save CSV."""
    RESULTS_DIR.mkdir(exist_ok=True)
    csv_path = RESULTS_DIR / f"{tech_name}_eval.csv"

    if not force and csv_path.exists():
        print(f"{tech_name}: eval already done ({csv_path}). Use --force.")
        return pd.read_csv(csv_path)

    model_dir = MODELS_DIR / tech_name

    # Check model exists
    has_model = any(
        (model_dir / f).exists()
        for f in ["final_model.zip", "final_model.pt", "final_model.npz",
                   "best_model.zip", "best_model.pt", "best_model.npz"]
    )
    if not has_model:
        print(f"{tech_name}: no model found in {model_dir}. Train first.")
        return None

    policy_fn = build_policy_fn(tech_name, tc, model_dir)

    seeds = config["evaluation"]["seeds"]
    modes = config["evaluation"]["scenario_modes"]

    records = []
    for mode in modes:
        for seed in seeds:
            env = IndustrialInventoryEnv(
                student_config, scenario_mode=mode, domain_randomization=True)
            obs, info = env.reset(seed=seed)

            done = False
            while not done:
                quantities = policy_fn(obs)
                # Convert quantities → action indices
                action_indices = np.array(
                    [q // 10 for q in quantities], dtype=np.int64)
                obs, reward, terminated, truncated, step_info = env.step(
                    action_indices)
                done = terminated or truncated

            costs = step_info["costs"]
            records.append({
                "technique": tech_name,
                "scenario_mode": mode,
                "seed": seed,
                "total_cost": costs["episode_total"],
                "holding_cost": costs["holding"],
                "stockout_cost": costs["stockout"],
                "ordering_cost": costs["ordering"],
                "discard_cost": costs["discarding"],
            })

    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False)

    # Print summary
    avg = df["total_cost"].mean()
    std = df["total_cost"].std()
    print(f"{tech_name}: Avg Cost = {avg:,.0f}  Std = {std:,.0f}  "
          f"({len(df)} episodes)")
    return df


def generate_comparison(config):
    """Build comparison.csv from individual eval CSVs."""
    rows = []
    for tech_name in config["techniques"]:
        csv_path = RESULTS_DIR / f"{tech_name}_eval.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            rows.append({
                "Technique": config["techniques"][tech_name].get(
                    "portal_name", tech_name),
                "Avg Cost": df["total_cost"].mean(),
                "Std Cost": df["total_cost"].std(),
                "Min Cost": df["total_cost"].min(),
                "Max Cost": df["total_cost"].max(),
                "Avg Holding": df["holding_cost"].mean(),
                "Avg Stockout": df["stockout_cost"].mean(),
                "Avg Ordering": df["ordering_cost"].mean(),
                "Avg Discard": df["discard_cost"].mean(),
            })

    if not rows:
        print("No evaluation results found.")
        return

    comp = pd.DataFrame(rows).sort_values("Avg Cost")
    comp.to_csv(RESULTS_DIR / "comparison.csv", index=False)

    print(f"\n{'='*80}")
    print("  Technique Comparison (sorted by Avg Cost)")
    print(f"{'='*80}")
    print(comp.to_string(index=False, float_format="%.0f"))
    print()

    # Best 5
    if len(comp) >= 5:
        best5 = comp.head(5)
        avg_of_best5 = best5["Avg Cost"].mean()
        print(f"Best 5 average: {avg_of_best5:,.0f}")
    return comp


def main():
    config = load_config()
    student_config = generate_student_config(config["roll_number"])
    all_techniques = list(config["techniques"].keys())

    parser = argparse.ArgumentParser(description="Evaluate RL policies")
    parser.add_argument("technique", choices=all_techniques + ["all"])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    techniques = all_techniques if args.technique == "all" else [args.technique]

    for tech in techniques:
        tc = config["techniques"][tech]
        evaluate_technique(tech, tc, config, student_config, args.force)

    if args.technique == "all":
        generate_comparison(config)


if __name__ == "__main__":
    main()
