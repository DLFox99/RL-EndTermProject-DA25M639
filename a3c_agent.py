"""A3C — Asynchronous Advantage Actor-Critic.

Multiple workers train in parallel, each with its own environment.
Gradients are pushed to a shared model asynchronously (Hogwild-style).

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

        advantages = returns - values_t.detach()

        # Losses
        policy_loss = -(log_probs_t * advantages).mean()
        value_loss = F.mse_loss(values_t, returns)
        entropy_loss = -entropies_t.mean()

        total_loss = policy_loss + 0.5 * value_loss + ent_coef * entropy_loss

        # Push gradients to the shared model. total_loss belongs to
        # local_model, so its gradients must be cleared every episode.
        local_model.zero_grad(set_to_none=True)
        shared_optimizer.zero_grad()

        total_loss.backward()
        nn.utils.clip_grad_norm_(local_model.parameters(), max_grad_norm)

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

        # Parent process owns rolling-cost / best-model tracking.
        result_queue.put((ep_num, float(ep_cost)))

        if ep_num % 500 == 0:
            print(f"  A3C worker {worker_id}: ep {ep_num:,}  cost={ep_cost:,.0f}")
