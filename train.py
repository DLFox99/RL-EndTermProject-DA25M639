#!/usr/bin/env python3
"""Train RL techniques for inventory control.

Usage:
    python train.py ppo              # train one (skip if done, resume if partial)
    python train.py all              # train all sequentially
    python train.py ppo --force      # retrain from scratch
    python train.py ppo --timesteps 2000000  # override total timesteps
"""

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).parent
MODELS_DIR = PROJECT_ROOT / "models"
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def get_student_config(config):
    from industrial_inventory_env import generate_student_config
    return generate_student_config(config["roll_number"])


# ---------------------------------------------------------------------------
# SB3 Callback — checkpointing + episode logging + best model tracking
# ---------------------------------------------------------------------------

from stable_baselines3.common.callbacks import BaseCallback


class PipelineCallback(BaseCallback):
    """Checkpoint every N minutes, log every episode, track best model."""

    def __init__(self, model_dir, checkpoint_min=5, steps_offset=0,
                 episodes_offset=0, best_cost=float("inf"), verbose=0):
        super().__init__(verbose)
        self.model_dir = Path(model_dir)
        self.checkpoint_sec = checkpoint_min * 60
        self.steps_offset = steps_offset
        self.episode_count = episodes_offset
        self.best_cost = best_cost
        self.recent_costs = []

        self.log_path = self.model_dir / "train_log.csv"
        self.start_wall = time.time()
        self.last_ckpt_wall = time.time()

        # Ensure CSV header exists
        if not self.log_path.exists() or self.log_path.stat().st_size == 0:
            with open(self.log_path, "w") as f:
                f.write("episode,timestep,episode_cost,wall_time_s,timestamp\n")

    @property
    def total_steps(self):
        return self.steps_offset + self.num_timesteps

    def _on_step(self) -> bool:
        # --- episode logging ---
        for info in self.locals.get("infos", []):
            if "episode" in info:
                ep_return = info["episode"]["r"]
                ep_cost = -ep_return * 100
                self.episode_count += 1
                self.recent_costs.append(ep_cost)

                with open(self.log_path, "a") as f:
                    f.write(f"{self.episode_count},{self.total_steps},"
                            f"{ep_cost:.2f},{time.time()-self.start_wall:.1f},"
                            f"{datetime.now().isoformat()}\n")

                # --- best model tracking (rolling 100) ---
                if len(self.recent_costs) >= 100:
                    roll_avg = np.mean(self.recent_costs[-100:])
                    if roll_avg < self.best_cost:
                        self.best_cost = roll_avg
                        self.model.save(str(self.model_dir / "best_model"))

        # --- time-based checkpoint ---
        now = time.time()
        if now - self.last_ckpt_wall >= self.checkpoint_sec:
            self._save_checkpoint()
            self.last_ckpt_wall = now

        return True

    def _save_checkpoint(self):
        ckpt = str(self.model_dir / "checkpoints" / "checkpoint")
        self.model.save(ckpt)

        # Replay buffer for off-policy
        if hasattr(self.model, "replay_buffer") and self.model.replay_buffer is not None:
            buf_path = str(self.model_dir / "checkpoints" / "replay_buffer")
            self.model.save_replay_buffer(buf_path)

        self._write_metadata()
        self._print_status()

    def _write_metadata(self):
        meta = {
            "steps_completed": self.total_steps,
            "episodes_completed": self.episode_count,
            "wall_time_s": time.time() - self.start_wall,
            "best_rolling_cost": self.best_cost
                if self.best_cost < float("inf") else None,
            "last_checkpoint": datetime.now().isoformat(),
        }
        with open(self.model_dir / "training_metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

    def _print_status(self):
        recent = self.recent_costs[-100:] if self.recent_costs else []
        avg = np.mean(recent) if recent else 0
        print(f"  [checkpoint] steps={self.total_steps:,}  "
              f"episodes={self.episode_count:,}  "
              f"recent_avg_cost={avg:,.0f}  "
              f"best={self.best_cost:,.0f}")

    def on_training_end(self):
        self._save_checkpoint()


# ---------------------------------------------------------------------------
# Environment factories
# ---------------------------------------------------------------------------

def make_flat_env(student_config):
    from industrial_inventory_env import IndustrialInventoryEnv
    from env_wrappers import FlattenObsWrapper
    from stable_baselines3.common.monitor import Monitor
    raw = IndustrialInventoryEnv(student_config, scenario_mode="random",
                                 domain_randomization=True)
    return Monitor(FlattenObsWrapper(raw))


def make_discrete_env(student_config):
    from industrial_inventory_env import IndustrialInventoryEnv
    from env_wrappers import FlattenObsWrapper, DiscreteActionWrapper
    from stable_baselines3.common.monitor import Monitor
    raw = IndustrialInventoryEnv(student_config, scenario_mode="random",
                                 domain_randomization=True)
    return Monitor(DiscreteActionWrapper(FlattenObsWrapper(raw)))


# ---------------------------------------------------------------------------
# Generic SB3 trainer
# ---------------------------------------------------------------------------

def _load_metadata(model_dir):
    meta_path = model_dir / "training_metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            return json.load(f)
    return None


def train_sb3(algo_class, tech_name, config, tc, model_dir, student_config, force):
    """Train any SB3 algorithm (on-policy or off-policy)."""
    from stable_baselines3.common.vec_env import DummyVecEnv

    total = tc["total_timesteps"]
    is_discrete = tc.get("action_type") == "discrete"
    is_offpolicy = tc.get("category") == "offpolicy"

    # --- skip if done ---
    meta = _load_metadata(model_dir)
    if not force and meta and meta.get("steps_completed", 0) >= total:
        print(f"{tech_name}: already at {meta['steps_completed']:,} steps "
              f"(target {total:,}). Use --force to retrain.")
        return

    # --- env factory ---
    env_fn = (lambda: make_discrete_env(student_config)) if is_discrete \
        else (lambda: make_flat_env(student_config))
    n_envs = tc.get("n_envs", 1)
    vec_env = DummyVecEnv([env_fn for _ in range(n_envs)])

    # --- resume or create ---
    ckpt_path = model_dir / "checkpoints" / "checkpoint.zip"
    steps_done = 0
    episodes_done = 0
    best_cost = float("inf")

    if not force and ckpt_path.exists():
        print(f"{tech_name}: resuming from checkpoint")
        model = algo_class.load(str(model_dir / "checkpoints" / "checkpoint"),
                                env=vec_env)
        if meta:
            steps_done = meta.get("steps_completed", 0)
            episodes_done = meta.get("episodes_completed", 0)
            best_cost = meta.get("best_rolling_cost") or float("inf")

        # Load replay buffer for off-policy
        if is_offpolicy:
            buf_path = model_dir / "checkpoints" / "replay_buffer.pkl"
            if buf_path.exists():
                model.load_replay_buffer(str(model_dir / "checkpoints" / "replay_buffer"))
                print(f"  Replay buffer loaded")
    else:
        if force and model_dir.exists():
            for f in model_dir.iterdir():
                if f.is_file():
                    f.unlink()
            ckpt_dir = model_dir / "checkpoints"
            if ckpt_dir.exists():
                shutil.rmtree(ckpt_dir)
            ckpt_dir.mkdir(parents=True)

        # Build kwargs from config
        kwargs = dict(
            learning_rate=tc["learning_rate"],
            gamma=tc.get("gamma", 0.99),
            verbose=1,
            policy_kwargs=dict(net_arch=tc["net_arch"]),
            seed=config.get("seed", 42),
        )

        # On-policy specific
        if not is_offpolicy:
            kwargs.update(
                n_steps=tc.get("n_steps", 2048),
                gae_lambda=tc.get("gae_lambda", 0.95),
                ent_coef=tc.get("ent_coef", 0.01),
            )
            if "batch_size" in tc:
                kwargs["batch_size"] = tc["batch_size"]
            if "n_epochs" in tc:
                kwargs["n_epochs"] = tc["n_epochs"]
            if "clip_range" in tc:
                kwargs["clip_range"] = tc["clip_range"]
        else:
            # Off-policy specific
            kwargs.update(
                buffer_size=tc.get("buffer_size", 100000),
                learning_starts=tc.get("learning_starts", 5000),
                batch_size=tc.get("batch_size", 128),
                tau=tc.get("tau", 0.005),
                exploration_fraction=tc.get("exploration_fraction", 0.3),
                exploration_initial_eps=tc.get("exploration_initial_eps", 1.0),
                exploration_final_eps=tc.get("exploration_final_eps", 0.05),
                train_freq=tc.get("train_freq", 4),
                gradient_steps=tc.get("gradient_steps", 1),
                target_update_interval=tc.get("target_update_interval", 1000),
            )

        model = algo_class("MlpPolicy", vec_env, **kwargs)

    # --- save hyperparams ---
    with open(model_dir / "hyperparams_used.yaml", "w") as f:
        yaml.dump(tc, f, default_flow_style=False)

    remaining = total - steps_done
    if remaining <= 0:
        print(f"{tech_name}: fully trained.")
        return

    print(f"{tech_name}: training {remaining:,} steps "
          f"({steps_done:,} → {total:,})")

    cb = PipelineCallback(
        model_dir=model_dir,
        checkpoint_min=config.get("checkpoint_interval_min", 5),
        steps_offset=steps_done,
        episodes_offset=episodes_done,
        best_cost=best_cost,
    )

    model.learn(total_timesteps=remaining, callback=cb)

    # --- save final model ---
    model.save(str(model_dir / "final_model"))
    cb._write_metadata()
    print(f"{tech_name}: done. Saved final_model.")


# ---------------------------------------------------------------------------
# REINFORCE trainer (custom loop)
# ---------------------------------------------------------------------------

def train_reinforce(config, tc, model_dir, student_config, force):
    import torch
    from reinforce_agent import ReinforcePolicy
    from env_wrappers import flatten_observation
    from industrial_inventory_env import IndustrialInventoryEnv

    total_episodes = tc["num_episodes"]
    lr = tc["learning_rate"]
    gamma = tc.get("gamma", 0.99)
    hidden = tc.get("hidden", 128)
    baseline_alpha = tc.get("baseline_alpha", 0.01)
    max_grad_norm = tc.get("max_grad_norm", 1.0)
    ckpt_sec = config.get("checkpoint_interval_min", 5) * 60

    # --- skip if done ---
    meta = _load_metadata(model_dir)
    if not force and meta and meta.get("episodes_completed", 0) >= total_episodes:
        print(f"REINFORCE: already at {meta['episodes_completed']:,} episodes. "
              f"Use --force to retrain.")
        return

    # --- resume or create ---
    ckpt_path = model_dir / "checkpoints" / "checkpoint.pt"
    ep_start = 0
    baseline = 0.0
    best_cost = float("inf")

    policy = ReinforcePolicy(hidden=hidden)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    if not force and ckpt_path.exists():
        print("REINFORCE: resuming from checkpoint")
        ckpt_data = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        policy.load_state_dict(ckpt_data["model_state"])
        optimizer.load_state_dict(ckpt_data["optimizer_state"])
        ep_start = ckpt_data.get("episode", 0)
        baseline = ckpt_data.get("baseline", 0.0)
        best_cost = ckpt_data.get("best_cost", float("inf"))
    elif force:
        for f in model_dir.iterdir():
            if f.is_file():
                f.unlink()
        ckpt_dir = model_dir / "checkpoints"
        if ckpt_dir.exists():
            shutil.rmtree(ckpt_dir)
        ckpt_dir.mkdir(parents=True)

    # --- save hyperparams ---
    with open(model_dir / "hyperparams_used.yaml", "w") as f:
        yaml.dump(tc, f, default_flow_style=False)

    if ep_start >= total_episodes:
        print("REINFORCE: fully trained.")
        return

    print(f"REINFORCE: training episodes {ep_start:,} → {total_episodes:,}")

    env = IndustrialInventoryEnv(student_config, scenario_mode="random",
                                  domain_randomization=True)

    log_path = model_dir / "train_log.csv"
    if not log_path.exists() or log_path.stat().st_size == 0:
        with open(log_path, "w") as f:
            f.write("episode,timestep,episode_cost,wall_time_s,timestamp\n")

    start_wall = time.time()
    last_ckpt_wall = start_wall
    recent_costs = []
    total_steps = ep_start * 50  # approximate

    for ep in range(ep_start + 1, total_episodes + 1):
        obs, _ = env.reset()
        saved_log_probs = []
        rewards = []
        done = False

        while not done:
            obs_flat = flatten_observation(obs)
            obs_t = torch.FloatTensor(obs_flat).unsqueeze(0)
            action_t, log_prob = policy.get_action(obs_t, deterministic=False)
            action_indices = action_t.squeeze(0).numpy().astype(np.int64)

            next_obs, reward, terminated, truncated, info = env.step(action_indices)
            saved_log_probs.append(log_prob.squeeze(0))
            rewards.append(reward)
            obs = next_obs
            done = terminated or truncated
            total_steps += 1

        # Compute returns
        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns_t = torch.FloatTensor(returns)
        episode_return = returns_t[0].item()
        episode_cost = -episode_return * 100

        # Update baseline
        baseline = baseline * (1 - baseline_alpha) + episode_return * baseline_alpha

        # Policy gradient
        advantages = returns_t - baseline
        log_prob_stack = torch.stack(saved_log_probs)
        loss = -(log_prob_stack * advantages.detach()).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=max_grad_norm)
        optimizer.step()

        # Log
        recent_costs.append(episode_cost)
        with open(log_path, "a") as f:
            f.write(f"{ep},{total_steps},{episode_cost:.2f},"
                    f"{time.time()-start_wall:.1f},{datetime.now().isoformat()}\n")

        # Best model
        if len(recent_costs) >= 100:
            roll_avg = np.mean(recent_costs[-100:])
            if roll_avg < best_cost:
                best_cost = roll_avg
                torch.save(policy.state_dict(),
                           str(model_dir / "best_model.pt"))

        # Time-based checkpoint
        now = time.time()
        if now - last_ckpt_wall >= ckpt_sec:
            torch.save({
                "model_state": policy.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "episode": ep,
                "baseline": baseline,
                "best_cost": best_cost,
            }, str(ckpt_path))

            meta = {
                "steps_completed": total_steps,
                "episodes_completed": ep,
                "wall_time_s": now - start_wall,
                "best_rolling_cost": best_cost if best_cost < float("inf") else None,
                "last_checkpoint": datetime.now().isoformat(),
            }
            with open(model_dir / "training_metadata.json", "w") as f:
                json.dump(meta, f, indent=2)

            avg = np.mean(recent_costs[-100:]) if recent_costs else 0
            print(f"  [checkpoint] ep={ep:,}  recent_avg_cost={avg:,.0f}  "
                  f"best={best_cost:,.0f}")
            last_ckpt_wall = now

        # Print progress
        if ep % 500 == 0:
            avg = np.mean(recent_costs[-500:]) if len(recent_costs) >= 500 \
                else np.mean(recent_costs)
            print(f"  Episode {ep:6,} | Avg Cost (last 500): {avg:,.0f}")

    # Save final
    torch.save(policy.state_dict(), str(model_dir / "final_model.pt"))
    meta = {
        "steps_completed": total_steps,
        "episodes_completed": total_episodes,
        "wall_time_s": time.time() - start_wall,
        "best_rolling_cost": best_cost if best_cost < float("inf") else None,
        "completed_at": datetime.now().isoformat(),
    }
    with open(model_dir / "training_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("REINFORCE: done. Saved final_model.pt")


# ---------------------------------------------------------------------------
# Technique registry
# ---------------------------------------------------------------------------

def _make_trainer(algo_name, category):
    """Create a training function for an SB3 algorithm."""
    def trainer(config, tc, model_dir, student_config, force):
        if algo_name == "DoubleDQN":
            from double_dqn import DoubleDQN as cls
        else:
            import stable_baselines3
            cls = getattr(stable_baselines3, algo_name)
        train_sb3(cls, tc.get("portal_name", algo_name),
                  config, tc, model_dir, student_config, force)
    return trainer


TRAINERS = {
    "ppo":       _make_trainer("PPO", "onpolicy"),
    "a2c":       _make_trainer("A2C", "onpolicy"),
    "dqn":       _make_trainer("DQN", "offpolicy"),
    "ddqn":      _make_trainer("DoubleDQN", "offpolicy"),
    "reinforce": train_reinforce,
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    all_techniques = list(TRAINERS.keys())
    parser = argparse.ArgumentParser(description="Train RL techniques")
    parser.add_argument("technique", choices=all_techniques + ["all"],
                        help="Technique to train, or 'all'")
    parser.add_argument("--force", action="store_true",
                        help="Delete existing model and retrain from scratch")
    parser.add_argument("--timesteps", type=int, default=None,
                        help="Override total_timesteps (or num_episodes for REINFORCE)")
    args = parser.parse_args()

    config = load_config()
    student_config = get_student_config(config)

    techniques = all_techniques if args.technique == "all" else [args.technique]

    for tech in techniques:
        model_dir = MODELS_DIR / tech
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "checkpoints").mkdir(exist_ok=True)

        tc = config["techniques"][tech].copy()

        # Override timesteps if requested
        if args.timesteps:
            if tech == "reinforce":
                tc["num_episodes"] = args.timesteps
            else:
                tc["total_timesteps"] = args.timesteps

        print(f"\n{'='*60}")
        print(f"  {tc.get('portal_name', tech)}")
        print(f"{'='*60}")

        TRAINERS[tech](config, tc, model_dir, student_config, args.force)


if __name__ == "__main__":
    main()
