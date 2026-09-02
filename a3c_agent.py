"""A3C — Asynchronous Advantage Actor-Critic.

Multiple workers collect rollouts in parallel, each with its own
environment. Gradient computation and rollout collection happen fully in
parallel across workers; the shared-model gradient assignment and
optimizer.step() are serialized under a lock to prevent gradient
corruption from concurrent writes to the shared parameters (see the
critical section in a3c_worker for details).

Policy is factored: 3 independent categorical heads (one per product).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    """Shared-trunk actor-critic with factored action heads."""

    def __init__(self, obs_dim=38, hidden=128, n_actions=11):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        # 3 actor heads (one per product)
        self.actor_heads = nn.ModuleList([
            nn.Linear(hidden, n_actions) for _ in range(3)
        ])
        # 1 critic head
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.shared(x)
        logits = [head(h) for head in self.actor_heads]
        value = self.critic(h)
        return logits, value

    def get_action(self, obs_tensor, deterministic=False):
        logits, value = self.forward(obs_tensor)
        actions, log_probs, entropies = [], [], []
        for logit in logits:
            dist = Categorical(logits=logit)
            if deterministic:
                a = logit.argmax(dim=-1)
            else:
                a = dist.sample()
            actions.append(a)
            log_probs.append(dist.log_prob(a))
            entropies.append(dist.entropy())
        return actions, log_probs, entropies, value


class SharedAdam(torch.optim.Adam):
    """Adam with shared state tensors for cross-process use."""

    def __init__(self, params, lr=1e-4):
        super().__init__(params, lr=lr)
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["step"] = torch.zeros(1)
                state["exp_avg"] = torch.zeros_like(p.data)
                state["exp_avg_sq"] = torch.zeros_like(p.data)
                state["step"].share_memory_()
                state["exp_avg"].share_memory_()
                state["exp_avg_sq"].share_memory_()


def a3c_worker(worker_id, shared_model, shared_optimizer, student_config,
               flatten_fn, n_steps, total_episodes, assignment_counter,
               completed_counter, lock, result_queue, log_path,
               gamma=0.99, ent_coef=0.01, max_grad_norm=1.0,
               hidden=128):
    """A3C worker process."""
    from industrial_inventory_env import IndustrialInventoryEnv

    env = IndustrialInventoryEnv(
        student_config, scenario_mode="random", domain_randomization=True)
    local_model = ActorCritic(hidden=hidden)

    while True:
        # Reserve a unique episode number. This counter represents
        # ASSIGNED work only; completed_counter is updated after the episode.
        with lock:
            current_ep = assignment_counter.value
            if current_ep >= total_episodes:
                break
            assignment_counter.value += 1
            ep_num = assignment_counter.value

        # Sync local model
        local_model.load_state_dict(shared_model.state_dict())

        obs, _ = env.reset()
        done = False
        values, log_probs_all, entropies_all, rewards = [], [], [], []

        while not done:
            obs_t = torch.FloatTensor(flatten_fn(obs)).unsqueeze(0)
            actions, log_probs, ents, value = local_model.get_action(obs_t)

            action_array = np.array([a.item() for a in actions], dtype=np.int64)
            next_obs, reward, terminated, truncated, info = env.step(action_array)
            done = terminated or truncated

            values.append(value)
            log_probs_all.append(torch.stack(log_probs).sum())
            entropies_all.append(torch.stack(ents).sum())
            rewards.append(reward)
            obs = next_obs

        # Compute returns and advantages
        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns)

        values_t = torch.cat(values).squeeze()
        log_probs_t = torch.stack(log_probs_all)
        entropies_t = torch.stack(entropies_all)

        # Normalize returns once, up front, and use this normalized version
        # everywhere downstream. Advantage normalization alone was not
        # sufficient: with raw returns (order 1e5-1e6) feeding value_loss,
        # and actor/critic sharing the same trunk, an exploding
        # value_loss gradient corrupted the actor's shared features even
        # once the actor's own loss term was well-scaled. Diagnostics
        # confirmed this -- value_loss climbed into the 1e7 range at
        # exactly the point entropy collapsed and cost reverted, despite
        # advantages already being standardized separately.
        #
        # values_t (the critic's raw output) must be compared against a
        # target on the SAME scale it is being trained to predict, so we
        # normalize returns first and derive both the value_loss target
        # and the advantage from that single normalized series. This
        # keeps the critic's regression target and the actor's advantage
        # signal internally consistent, rather than mixing a raw-scale
        # critic prediction against a normalized target.
        returns_norm = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-8)

        advantages = returns_norm - values_t.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        # Losses
        policy_loss = -(log_probs_t * advantages).mean()
        value_loss = F.mse_loss(values_t, returns_norm)
        entropy_loss = -entropies_t.mean()

        total_loss = policy_loss + 0.5 * value_loss + ent_coef * entropy_loss

        # Push gradients to the shared model. total_loss belongs to
        # local_model, so its gradients must be cleared every episode.
        local_model.zero_grad(set_to_none=True)
        shared_optimizer.zero_grad()

        total_loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(
            local_model.parameters(), max_grad_norm)

        # CRITICAL SECTION: assigning this worker's gradient into the
        # shared model's .grad tensors and calling optimizer.step() must
        # happen atomically. Without the lock, another worker can
        # overwrite shared_p._grad between the assignment loop and the
        # step() call, causing the optimizer to apply a mismatched or
        # partially-overwritten gradient — this was corrupting training
        # (cost would dip early then diverge as corrupted updates
        # accumulated). Serializing this section trades some of A3C's
        # nominal parallelism for correctness; rollout collection (the
        # expensive part) still happens fully in parallel across workers,
        # only this brief update step is serialized.
        with lock:
            for local_p, shared_p in zip(local_model.parameters(),
                                          shared_model.parameters()):
                if local_p.grad is None:
                    shared_p._grad = None
                else:
                    shared_p._grad = local_p.grad.detach().clone()

            shared_optimizer.step()

        # Log episode cost
        ep_cost = sum(-r * 100 for r in rewards)
        try:
            # Serialize one complete CSV-line write across workers.
            with lock:
                with open(log_path, "a") as f:
                    f.write(f"{ep_num},{ep_num * 50},{ep_cost:.2f},"
                            f"0,worker_{worker_id}\n")
        except Exception:
            pass  # don't crash worker on logging failure

        # Mark completion only after rollout + shared-model update finish.
        with lock:
            completed_counter.value += 1

        # Parent process owns rolling-cost / best-model tracking and writes
        # diagnostics so worker processes never contend on diagnostics.csv.
        result_queue.put((
            ep_num,
            float(ep_cost),
            {
                "worker_id": int(worker_id),
                "policy_loss": float(policy_loss.detach().item()),
                "value_loss": float(value_loss.detach().item()),
                "entropy": float(entropies_t.mean().detach().item()),
                "total_loss": float(total_loss.detach().item()),
                "advantage_mean": float(advantages.mean().detach().item()),
                "advantage_std": float(
                    advantages.std(unbiased=False).detach().item()),
                "grad_norm": float(
                    grad_norm.detach().item()
                    if hasattr(grad_norm, "detach") else grad_norm),
                "worker_return": float(sum(rewards)),
            },
        ))

        if ep_num % 500 == 0:
            print(f"  A3C worker {worker_id}: ep {ep_num:,}  cost={ep_cost:,.0f}")
