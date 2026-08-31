#!/usr/bin/env python3
"""Train RL techniques for inventory control.

Usage:
    python train.py ppo              # train one (skip if done, resume if partial)
    python train.py all              # train all sequentially
    python train.py ppo --force      # retrain from scratch
    python train.py ppo --timesteps 2000000  # override total timesteps
    python train.py reinforce --episodes 200000  # override episode budget
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

import wandb_utils

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
    """Checkpoint, log episodes, track best model, and report convergence.

    Convergence is diagnostic only: it never stops training.  The configured
    timestep budget remains authoritative.
    """

    def __init__(self, model_dir, checkpoint_min=5, steps_offset=0,
                 episodes_offset=0, best_cost=float("inf"),
                 technique_name="SB3", convergence_state=None, verbose=0):
        super().__init__(verbose)
        self.model_dir = Path(model_dir)
        self.checkpoint_sec = checkpoint_min * 60
        self.steps_offset = steps_offset
        self.episode_count = episodes_offset
        self.best_cost = best_cost
        self.technique_name = technique_name
        self.recent_costs = []
        self.convergence_state = convergence_state or {
            "detected": False,
            "first_step": None,
            "first_episode": None,
            "reason": None,
        }

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

    @property
    def converged(self):
        """Whether convergence has ever been detected during this run."""
        return bool(self.convergence_state["detected"])

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

                # --- wandb logging ---
                roll_avg = np.mean(self.recent_costs[-100:]) \
                    if len(self.recent_costs) >= 100 else np.mean(self.recent_costs)
                wandb_utils.log_episode(
                    self.episode_count, self.total_steps, ep_cost,
                    rolling_avg=roll_avg, best_cost=self.best_cost)

        # --- time-based checkpoint ---
        now = time.time()
        if now - self.last_ckpt_wall >= self.checkpoint_sec:
            self._check_convergence()
            self._save_checkpoint()
            self.last_ckpt_wall = now

        # Convergence is advisory only.  Returning True keeps SB3 training
        # until the requested timestep budget is exhausted.
        return True

    def _check_convergence(self):
        _check_convergence_warning(
            log_path=self.log_path,
            label=self.technique_name,
            step=self.total_steps,
            episode=self.episode_count,
            state=self.convergence_state,
        )

    def _save_checkpoint(self):
        ckpt = str(self.model_dir / "checkpoints" / "checkpoint")
        self.model.save(ckpt)

        # Replay buffer for off-policy
        if hasattr(self.model, "replay_buffer") and self.model.replay_buffer is not None:
            buf_path = str(self.model_dir / "checkpoints" / "replay_buffer")
            self.model.save_replay_buffer(buf_path)

        self._write_metadata()
        self._print_status()
        wandb_utils.log_checkpoint(
            self.total_steps, self.episode_count,
            self.best_cost, self.converged)

    def _write_metadata(self):
        meta = {
            "steps_completed": self.total_steps,
            "episodes_completed": self.episode_count,
            "wall_time_s": time.time() - self.start_wall,
            "best_rolling_cost": self.best_cost
                if self.best_cost < float("inf") else None,
            "convergence_detected": self.convergence_state["detected"],
            "convergence_first_step": self.convergence_state["first_step"],
            "convergence_first_episode": self.convergence_state["first_episode"],
            "convergence_reason": self.convergence_state["reason"],
            "last_checkpoint": datetime.now().isoformat(),
        }
        with open(self.model_dir / "training_metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

    def _print_status(self):
        recent = self.recent_costs[-100:] if self.recent_costs else []
        avg = np.mean(recent) if recent else 0
        convergence_note = "  convergence=detected" if self.converged else ""
        print(f"  [checkpoint] steps={self.total_steps:,}  "
              f"episodes={self.episode_count:,}  "
              f"recent_avg_cost={avg:,.0f}  "
              f"best={self.best_cost:,.0f}{convergence_note}")

    def on_training_end(self):
        self._check_convergence()
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


def _convergence_state_from_meta(meta=None):
    """Restore sticky, diagnostic convergence information from metadata."""
    meta = meta or {}
    return {
        "detected": bool(meta.get("convergence_detected", False)),
        "first_step": meta.get("convergence_first_step"),
        "first_episode": meta.get("convergence_first_episode"),
        "reason": meta.get("convergence_reason"),
    }


def _add_convergence_metadata(meta, state):
    """Add convergence diagnostics to a training metadata dictionary."""
    meta.update({
        "convergence_detected": state["detected"],
        "convergence_first_step": state["first_step"],
        "convergence_first_episode": state["first_episode"],
        "convergence_reason": state["reason"],
    })
    return meta


def _check_convergence_warning(log_path, label, step, episode, state):
    """Check convergence, warn once, and never alter the training budget."""
    from check_convergence import check_convergence

    result = check_convergence(log_path)
    if result["converged"] and not state["detected"]:
        state["detected"] = True
        state["first_step"] = int(step)
        state["first_episode"] = int(episode)
        state["reason"] = result["reason"]

        print()
        print(f"  *** CONVERGENCE WARNING [{label}]")
        print(f"  *** Detected at step {int(step):,}, episode {int(episode):,}")
        print(f"  *** {result['reason']}")
        print("  *** Training WILL CONTINUE to the configured target.")

    return result


def train_sb3(algo_class, tech_name, config, tc, model_dir, student_config, force):
    """Train any SB3 algorithm (on-policy or off-policy)."""
    from stable_baselines3.common.vec_env import DummyVecEnv

    total = tc["total_timesteps"]
    is_discrete = tc.get("action_type") == "discrete"
    is_offpolicy = tc.get("category") == "offpolicy"

    # --- wandb init ---
    wandb_utils.init(tech_name, config, tc)

    # --- skip if done ---
    meta = _load_metadata(model_dir)
    convergence_state = _convergence_state_from_meta(
        meta if not force else None)
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
    # Resume priority: checkpoints/checkpoint.zip (most recent, mid-training)
    #                   > final_model.zip (a previously completed/migrated run)
    #                   > fresh model
    ckpt_path = model_dir / "checkpoints" / "checkpoint.zip"
    final_path = model_dir / "final_model.zip"
    steps_done = 0
    episodes_done = 0
    best_cost = float("inf")

    resume_from = None
    if not force and ckpt_path.exists():
        resume_from = model_dir / "checkpoints" / "checkpoint"
        resume_kind = "checkpoint"
    elif not force and final_path.exists():
        resume_from = model_dir / "final_model"
        resume_kind = "final_model"

    if resume_from is not None:
        print(f"{tech_name}: resuming from {resume_kind} "
              f"({resume_from.name}.zip)")
        model = algo_class.load(str(resume_from), env=vec_env)
        if meta:
            steps_done = meta.get("steps_completed", 0)
            episodes_done = meta.get("episodes_completed", 0)
            best_cost = meta.get("best_rolling_cost") or float("inf")
        else:
            print(f"  WARNING: no training_metadata.json found for "
                  f"{tech_name}. Assuming 0 steps completed — if this "
                  f"model was actually trained, timestep counts in "
                  f"train_log.csv will be wrong. Run migrate_existing.py "
                  f"or create training_metadata.json manually.")

        # Load replay buffer for off-policy (only meaningful if resuming
        # from a checkpoint — a bare final_model won't have one)
        if is_offpolicy and resume_kind == "checkpoint":
            buf_path = model_dir / "checkpoints" / "replay_buffer.pkl"
            if buf_path.exists():
                model.load_replay_buffer(str(model_dir / "checkpoints" / "replay_buffer"))
                print(f"  Replay buffer loaded")
        elif is_offpolicy and resume_kind == "final_model":
            print(f"  Note: resuming off-policy algo from final_model with "
                  f"no replay buffer — buffer will rebuild from scratch "
                  f"during learning_starts warmup.")
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
        technique_name=tech_name,
        convergence_state=convergence_state,
    )

    model.learn(total_timesteps=remaining, callback=cb)

    # --- save final model ---
    model.save(str(model_dir / "final_model"))
    cb._write_metadata()
    if cb.converged:
        print(f"{tech_name}: done. Saved final_model. "
              f"Convergence was detected during training; "
              f"training still continued to the requested budget.")
    else:
        print(f"{tech_name}: done. Saved final_model.")
    wandb_utils.finish()


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

    # --- wandb init ---
    wandb_utils.init("reinforce", config, tc)

    # --- skip if done ---
    meta = _load_metadata(model_dir)
    convergence_state = _convergence_state_from_meta(
        meta if not force else None)
    if not force and meta and meta.get("episodes_completed", 0) >= total_episodes:
        print(f"REINFORCE: already at {meta['episodes_completed']:,} episodes. "
              f"Use --force to retrain.")
        return

    # --- resume or create ---
    # Resume priority: checkpoints/checkpoint.pt (has optimizer + baseline state)
    #                   > final_model.pt (weights only — a previous/migrated run)
    #                   > fresh model
    ckpt_path = model_dir / "checkpoints" / "checkpoint.pt"
    final_path = model_dir / "final_model.pt"
    ep_start = 0
    baseline = 0.0
    best_cost = float("inf")

    policy = ReinforcePolicy(hidden=hidden)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    if not force and ckpt_path.exists():
        print("REINFORCE: resuming from checkpoint (full state)")
        ckpt_data = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        policy.load_state_dict(ckpt_data["model_state"])
        optimizer.load_state_dict(ckpt_data["optimizer_state"])
        ep_start = ckpt_data.get("episode", 0)
        baseline = ckpt_data.get("baseline", 0.0)
        best_cost = ckpt_data.get("best_cost", float("inf"))
    elif not force and final_path.exists():
        print("REINFORCE: resuming from final_model.pt (weights only — "
              "optimizer momentum and baseline reset)")
        policy.load_state_dict(
            torch.load(str(final_path), map_location="cpu", weights_only=True))
        meta = _load_metadata(model_dir)
        if meta:
            ep_start = meta.get("episodes_completed", 0)
            best_cost = meta.get("best_rolling_cost") or float("inf")
        else:
            print("  WARNING: no training_metadata.json found. Assuming "
                  "0 episodes completed — episode/timestep counts in "
                  "train_log.csv will be wrong. Run migrate_existing.py "
                  "or create training_metadata.json manually.")
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

        roll_avg = np.mean(recent_costs[-100:]) \
            if len(recent_costs) >= 100 else np.mean(recent_costs)
        wandb_utils.log_episode(ep, total_steps, episode_cost,
                                rolling_avg=roll_avg, best_cost=best_cost)

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
            _check_convergence_warning(
                log_path=log_path,
                label="REINFORCE",
                step=total_steps,
                episode=ep,
                state=convergence_state,
            )
            meta = _add_convergence_metadata(meta, convergence_state)
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
    _check_convergence_warning(
        log_path=log_path,
        label="REINFORCE",
        step=total_steps,
        episode=total_episodes,
        state=convergence_state,
    )
    meta = {
        "steps_completed": total_steps,
        "episodes_completed": total_episodes,
        "wall_time_s": time.time() - start_wall,
        "best_rolling_cost": best_cost if best_cost < float("inf") else None,
        "completed_at": datetime.now().isoformat(),
    }
    meta = _add_convergence_metadata(meta, convergence_state)
    with open(model_dir / "training_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("REINFORCE: done. Saved final_model.pt")
    wandb_utils.finish()


# ---------------------------------------------------------------------------
# Tabular trainer (Q-Learning, SARSA, TD(λ))
# ---------------------------------------------------------------------------

def train_tabular(config, tc, model_dir, student_config, force):
    from tabular_agent import StateDiscretizer, TabularAgent
    from tabular_agent import (train_tabular_qlearning, train_tabular_sarsa,
                               train_td_lambda)
    from industrial_inventory_env import IndustrialInventoryEnv

    update_rule = tc["update_rule"]
    total_episodes = tc["num_episodes"]
    ckpt_sec = config.get("checkpoint_interval_min", 5) * 60
    name = tc.get("portal_name", update_rule)

    # --- wandb init ---
    wandb_utils.init(tc.get("update_rule", "tabular"), config, tc)

    meta = _load_metadata(model_dir)
    convergence_state = _convergence_state_from_meta(
        meta if not force else None)
    if not force and meta and meta.get("episodes_completed", 0) >= total_episodes:
        print(f"{name}: already at {meta['episodes_completed']:,} episodes.")
        return

    discretizer = StateDiscretizer()
    agent = TabularAgent(discretizer.n_states)

    ckpt_path = model_dir / "checkpoints" / "checkpoint.npz"
    final_path = model_dir / "final_model.npz"
    ep_start = 0
    best_cost = float("inf")

    if not force and ckpt_path.exists():
        print(f"{name}: resuming from checkpoint")
        agent.load(str(ckpt_path))
        if meta:
            ep_start = meta.get("episodes_completed", 0)
            best_cost = meta.get("best_rolling_cost") or float("inf")
    elif not force and final_path.exists():
        print(f"{name}: resuming from final_model.npz")
        agent.load(str(final_path))
        if meta:
            ep_start = meta.get("episodes_completed", 0)
            best_cost = meta.get("best_rolling_cost") or float("inf")
        else:
            print("  WARNING: no training_metadata.json found. Assuming "
                  "0 episodes completed.")
    elif force:
        for f in model_dir.iterdir():
            if f.is_file(): f.unlink()
        (model_dir / "checkpoints").mkdir(exist_ok=True)

    with open(model_dir / "hyperparams_used.yaml", "w") as f:
        yaml.dump(tc, f, default_flow_style=False)

    remaining = total_episodes - ep_start
    if remaining <= 0:
        print(f"{name}: fully trained.")
        return

    print(f"{name}: training episodes {ep_start:,} → {total_episodes:,}")

    env = IndustrialInventoryEnv(student_config, scenario_mode="random",
                                  domain_randomization=True)

    log_path = model_dir / "train_log.csv"
    if not log_path.exists() or log_path.stat().st_size == 0:
        with open(log_path, "w") as f:
            f.write("episode,timestep,episode_cost,wall_time_s,timestamp\n")

    # Save discretizer config for policy generation
    import json as _json
    with open(model_dir / "discretizer_config.json", "w") as f:
        _json.dump(discretizer.config_dict(), f)

    kwargs = dict(
        env=env, discretizer=discretizer, agent=agent,
        num_episodes=remaining,
        alpha=tc.get("alpha", 0.1), gamma=tc.get("gamma", 0.99),
        epsilon_start=tc.get("epsilon_start", 1.0),
        epsilon_end=tc.get("epsilon_end", 0.05),
    )

    if update_rule == "qlearning":
        gen = train_tabular_qlearning(**kwargs)
    elif update_rule == "sarsa":
        gen = train_tabular_sarsa(**kwargs)
    elif update_rule == "td_lambda":
        kwargs["lambd"] = tc.get("lambd", 0.8)
        gen = train_td_lambda(**kwargs)

    start_wall = time.time()
    last_ckpt_wall = start_wall
    recent_costs = []

    for rel_ep, ep_cost in gen:
        abs_ep = ep_start + rel_ep + 1
        recent_costs.append(ep_cost)

        with open(log_path, "a") as f:
            f.write(f"{abs_ep},{abs_ep*50},{ep_cost:.2f},"
                    f"{time.time()-start_wall:.1f},{datetime.now().isoformat()}\n")

        roll_avg = np.mean(recent_costs[-100:]) \
            if len(recent_costs) >= 100 else np.mean(recent_costs)
        wandb_utils.log_episode(abs_ep, abs_ep * 50, ep_cost,
                                rolling_avg=roll_avg, best_cost=best_cost)

        if len(recent_costs) >= 100:
            roll_avg = np.mean(recent_costs[-100:])
            if roll_avg < best_cost:
                best_cost = roll_avg
                agent.save(str(model_dir / "best_model.npz"))

        now = time.time()
        if now - last_ckpt_wall >= ckpt_sec:
            agent.save(str(ckpt_path))
            meta = {"steps_completed": abs_ep * 50, "episodes_completed": abs_ep,
                    "wall_time_s": now - start_wall,
                    "best_rolling_cost": best_cost if best_cost < float("inf") else None,
                    "last_checkpoint": datetime.now().isoformat()}
            _check_convergence_warning(
                log_path=log_path,
                label=name,
                step=abs_ep * 50,
                episode=abs_ep,
                state=convergence_state,
            )
            meta = _add_convergence_metadata(meta, convergence_state)
            with open(model_dir / "training_metadata.json", "w") as f:
                json.dump(meta, f, indent=2)
            avg = np.mean(recent_costs[-100:]) if recent_costs else 0
            print(f"  [checkpoint] ep={abs_ep:,}  avg={avg:,.0f}  best={best_cost:,.0f}")
            last_ckpt_wall = now

        if abs_ep % 2000 == 0:
            avg = np.mean(recent_costs[-2000:]) if len(recent_costs) >= 2000 \
                else np.mean(recent_costs)
            print(f"  Episode {abs_ep:6,} | Avg Cost: {avg:,.0f}")

    agent.save(str(model_dir / "final_model.npz"))
    _check_convergence_warning(
        log_path=log_path,
        label=name,
        step=total_episodes * 50,
        episode=total_episodes,
        state=convergence_state,
    )
    meta = {"steps_completed": total_episodes * 50, "episodes_completed": total_episodes,
            "wall_time_s": time.time() - start_wall,
            "best_rolling_cost": best_cost if best_cost < float("inf") else None,
            "completed_at": datetime.now().isoformat()}
    meta = _add_convergence_metadata(meta, convergence_state)
    with open(model_dir / "training_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"{name}: done. Saved final_model.npz")
    wandb_utils.finish()


# ---------------------------------------------------------------------------
# NN Q-Learning / NN SARSA trainer
# ---------------------------------------------------------------------------

def train_nn_custom(config, tc, model_dir, student_config, force):
    import torch
    from nn_agent import QNetworkFactored, train_nn_qlearning_episode, train_nn_sarsa_episode
    from env_wrappers import flatten_observation
    from industrial_inventory_env import IndustrialInventoryEnv

    update_rule = tc["update_rule"]
    total_episodes = tc["num_episodes"]
    lr = tc["learning_rate"]
    gamma = tc.get("gamma", 0.99)
    hidden = tc.get("hidden", 128)
    ckpt_sec = config.get("checkpoint_interval_min", 5) * 60
    name = tc.get("portal_name", update_rule)

    # --- wandb init ---
    wandb_utils.init(tc.get("update_rule", "nn_custom"), config, tc)

    meta = _load_metadata(model_dir)
    convergence_state = _convergence_state_from_meta(
        meta if not force else None)
    if not force and meta and meta.get("episodes_completed", 0) >= total_episodes:
        print(f"{name}: already at {meta['episodes_completed']:,} episodes.")
        return

    model = QNetworkFactored(hidden=hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    ckpt_path = model_dir / "checkpoints" / "checkpoint.pt"
    final_path = model_dir / "final_model.pt"
    ep_start = 0
    best_cost = float("inf")

    if not force and ckpt_path.exists():
        print(f"{name}: resuming from checkpoint")
        ckpt_data = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt_data["model_state"])
        optimizer.load_state_dict(ckpt_data["optimizer_state"])
        ep_start = ckpt_data.get("episode", 0)
        best_cost = ckpt_data.get("best_cost", float("inf"))
    elif not force and final_path.exists():
        print(f"{name}: resuming from final_model.pt (optimizer state reset)")
        model.load_state_dict(
            torch.load(str(final_path), map_location="cpu", weights_only=True))
        if meta:
            ep_start = meta.get("episodes_completed", 0)
            best_cost = meta.get("best_rolling_cost") or float("inf")
        else:
            print("  WARNING: no training_metadata.json found. Assuming "
                  "0 episodes completed.")
    elif force:
        for f in model_dir.iterdir():
            if f.is_file(): f.unlink()
        (model_dir / "checkpoints").mkdir(exist_ok=True)

    with open(model_dir / "hyperparams_used.yaml", "w") as f:
        yaml.dump(tc, f, default_flow_style=False)

    if ep_start >= total_episodes:
        print(f"{name}: fully trained.")
        return

    print(f"{name}: training episodes {ep_start:,} → {total_episodes:,}")

    env = IndustrialInventoryEnv(student_config, scenario_mode="random",
                                  domain_randomization=True)

    log_path = model_dir / "train_log.csv"
    if not log_path.exists() or log_path.stat().st_size == 0:
        with open(log_path, "w") as f:
            f.write("episode,timestep,episode_cost,wall_time_s,timestamp\n")

    train_fn = train_nn_qlearning_episode if update_rule == "qlearning" \
        else train_nn_sarsa_episode

    eps_start = tc.get("epsilon_start", 1.0)
    eps_end = tc.get("epsilon_end", 0.05)
    eps_decay = int(total_episodes * 0.7)

    start_wall = time.time()
    last_ckpt_wall = start_wall
    recent_costs = []

    for ep in range(ep_start + 1, total_episodes + 1):
        epsilon = max(eps_end, eps_start - (eps_start - eps_end) * ep / eps_decay)
        ep_cost = train_fn(env, model, optimizer, flatten_observation,
                           gamma=gamma, epsilon=epsilon)
        recent_costs.append(ep_cost)

        with open(log_path, "a") as f:
            f.write(f"{ep},{ep*50},{ep_cost:.2f},"
                    f"{time.time()-start_wall:.1f},{datetime.now().isoformat()}\n")

        roll_avg = np.mean(recent_costs[-100:]) \
            if len(recent_costs) >= 100 else np.mean(recent_costs)
        wandb_utils.log_episode(ep, ep * 50, ep_cost,
                                rolling_avg=roll_avg, best_cost=best_cost)

        if len(recent_costs) >= 100:
            roll_avg = np.mean(recent_costs[-100:])
            if roll_avg < best_cost:
                best_cost = roll_avg
                torch.save(model.state_dict(), str(model_dir / "best_model.pt"))

        now = time.time()
        if now - last_ckpt_wall >= ckpt_sec:
            torch.save({"model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "episode": ep, "best_cost": best_cost}, str(ckpt_path))
            meta = {"steps_completed": ep * 50, "episodes_completed": ep,
                    "wall_time_s": now - start_wall,
                    "best_rolling_cost": best_cost if best_cost < float("inf") else None,
                    "last_checkpoint": datetime.now().isoformat()}
            _check_convergence_warning(
                log_path=log_path,
                label=name,
                step=ep * 50,
                episode=ep,
                state=convergence_state,
            )
            meta = _add_convergence_metadata(meta, convergence_state)
            with open(model_dir / "training_metadata.json", "w") as f:
                json.dump(meta, f, indent=2)
            avg = np.mean(recent_costs[-100:]) if recent_costs else 0
            print(f"  [checkpoint] ep={ep:,}  avg={avg:,.0f}  best={best_cost:,.0f}")
            last_ckpt_wall = now

        if ep % 500 == 0:
            avg = np.mean(recent_costs[-500:]) if len(recent_costs) >= 500 \
                else np.mean(recent_costs)
            print(f"  Episode {ep:6,} | Avg Cost: {avg:,.0f}")

    torch.save(model.state_dict(), str(model_dir / "final_model.pt"))
    _check_convergence_warning(
        log_path=log_path,
        label=name,
        step=total_episodes * 50,
        episode=total_episodes,
        state=convergence_state,
    )
    meta = {"steps_completed": total_episodes * 50, "episodes_completed": total_episodes,
            "wall_time_s": time.time() - start_wall,
            "best_rolling_cost": best_cost if best_cost < float("inf") else None,
            "completed_at": datetime.now().isoformat()}
    meta = _add_convergence_metadata(meta, convergence_state)
    with open(model_dir / "training_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"{name}: done. Saved final_model.pt")
    wandb_utils.finish()


# ---------------------------------------------------------------------------
# A3C trainer
# ---------------------------------------------------------------------------

def train_a3c(config, tc, model_dir, student_config, force):
    import torch
    import torch.multiprocessing as mp
    from a3c_agent import ActorCritic, SharedAdam, a3c_worker
    from env_wrappers import flatten_observation

    total_episodes = tc["num_episodes"]
    n_workers = tc.get("n_workers", 4)
    hidden = tc.get("hidden", 128)
    name = tc.get("portal_name", "A3C")
    ckpt_sec = config.get("checkpoint_interval_min", 5) * 60

    # --- wandb init ---
    wandb_utils.init("a3c", config, tc)

    meta = _load_metadata(model_dir)
    convergence_state = _convergence_state_from_meta(
        meta if not force else None)
    if not force and meta and meta.get("episodes_completed", 0) >= total_episodes:
        print(f"{name}: already at {meta['episodes_completed']:,} episodes.")
        return

    if force:
        for f in model_dir.iterdir():
            if f.is_file(): f.unlink()
        (model_dir / "checkpoints").mkdir(exist_ok=True)

    with open(model_dir / "hyperparams_used.yaml", "w") as f:
        yaml.dump(tc, f, default_flow_style=False)

    log_path = model_dir / "train_log.csv"
    if not log_path.exists() or log_path.stat().st_size == 0:
        with open(log_path, "w") as f:
            f.write("episode,timestep,episode_cost,wall_time_s,timestamp\n")

    ep_start = 0 if force else (meta.get("episodes_completed", 0) if meta else 0)
    remaining = total_episodes - ep_start
    if remaining <= 0:
        print(f"{name}: fully trained.")
        return

    print(f"{name}: training {remaining:,} episodes with {n_workers} workers")

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    shared_model = ActorCritic(hidden=hidden)
    ckpt_path = model_dir / "checkpoints" / "checkpoint.pt"
    if not force and ckpt_path.exists():
        shared_model.load_state_dict(
            torch.load(str(ckpt_path), map_location="cpu", weights_only=True))
        print(f"{name}: loaded checkpoint")
    shared_model.share_memory()

    shared_optimizer = SharedAdam(shared_model.parameters(),
                                  lr=tc.get("learning_rate", 1e-4))
    global_counter = mp.Value("i", ep_start)
    lock = mp.Lock()

    start_wall = time.time()
    last_check_wall = start_wall
    workers = []
    for i in range(n_workers):
        p = mp.Process(target=a3c_worker, args=(
            i, shared_model, shared_optimizer, student_config,
            flatten_observation, 50, total_episodes, global_counter, lock,
            str(log_path), tc.get("gamma", 0.99), tc.get("ent_coef", 0.01),
            tc.get("max_grad_norm", 1.0), hidden))
        p.start()
        workers.append(p)

    # Monitor the shared log while workers train.  Convergence is advisory
    # only; workers are never stopped by this check.
    while any(p.is_alive() for p in workers):
        for p in workers:
            p.join(timeout=0.25)

        now = time.time()
        if now - last_check_wall >= ckpt_sec:
            with lock:
                current_ep = global_counter.value
            _check_convergence_warning(
                log_path=log_path,
                label=name,
                step=current_ep * 50,
                episode=current_ep,
                state=convergence_state,
            )
            last_check_wall = now

    for p in workers:
        p.join()

    # Save final model
    torch.save(shared_model.state_dict(), str(model_dir / "final_model.pt"))
    torch.save(shared_model.state_dict(), str(ckpt_path))

    _check_convergence_warning(
        log_path=log_path,
        label=name,
        step=total_episodes * 50,
        episode=total_episodes,
        state=convergence_state,
    )
    meta = {"steps_completed": total_episodes * 50,
            "episodes_completed": total_episodes,
            "wall_time_s": time.time() - start_wall,
            "best_rolling_cost": None,
            "completed_at": datetime.now().isoformat()}
    meta = _add_convergence_metadata(meta, convergence_state)
    with open(model_dir / "training_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"{name}: done. Saved final_model.pt")
    wandb_utils.finish()


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
    "ppo":               _make_trainer("PPO", "onpolicy"),
    "a2c":               _make_trainer("A2C", "onpolicy"),
    "dqn":               _make_trainer("DQN", "offpolicy"),
    "ddqn":              _make_trainer("DoubleDQN", "offpolicy"),
    "reinforce":         train_reinforce,
    "a3c":               train_a3c,
    "tabular_qlearning": train_tabular,
    "tabular_sarsa":     train_tabular,
    "td_lambda":         train_tabular,
    "nn_qlearning":      train_nn_custom,
    "nn_sarsa":          train_nn_custom,
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
    budget_group = parser.add_mutually_exclusive_group()
    budget_group.add_argument(
        "--timesteps", type=int, default=None,
        help="Override total_timesteps for timestep-based algorithms")
    budget_group.add_argument(
        "--episodes", type=int, default=None,
        help="Override num_episodes for episode-based algorithms")
    args = parser.parse_args()

    if args.technique == "all" and (
            args.timesteps is not None or args.episodes is not None):
        parser.error("--timesteps/--episodes cannot be used with 'all'; "
                     "set each technique budget in config.yaml instead")

    config = load_config()
    student_config = get_student_config(config)

    techniques = all_techniques if args.technique == "all" else [args.technique]

    for tech in techniques:
        model_dir = MODELS_DIR / tech
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "checkpoints").mkdir(exist_ok=True)

        tc = config["techniques"][tech].copy()

        if args.timesteps is not None:
            if "total_timesteps" not in tc:
                parser.error(f"{tech} is episode-based; use --episodes instead")
            tc["total_timesteps"] = args.timesteps

        if args.episodes is not None:
            if "num_episodes" not in tc:
                parser.error(f"{tech} is timestep-based; use --timesteps instead")
            tc["num_episodes"] = args.episodes

        print(f"\n{'='*60}")
        print(f"  {tc.get('portal_name', tech)}")
        print(f"{'='*60}")

        TRAINERS[tech](config, tc, model_dir, student_config, args.force)


if __name__ == "__main__":
    main()
