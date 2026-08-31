"""REINFORCE with baseline — manual PyTorch implementation.

Policy network has a shared trunk + 3 independent categorical heads (one per
product, each outputting logits over 11 discrete order levels 0..10).

Baseline: running mean of returns (simple, effective, avoids a second network).
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


class ReinforcePolicy(nn.Module):
    """Policy network with factored action heads."""

    def __init__(self, obs_dim: int = 38, hidden: int = 128, n_actions: int = 11):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.heads = nn.ModuleList([nn.Linear(hidden, n_actions) for _ in range(3)])

    def forward(self, x: torch.Tensor):
        h = self.shared(x)
        return [head(h) for head in self.heads]

    def get_action(self, obs: torch.Tensor, deterministic: bool = False):
        """Return (actions [3], log_prob scalar)."""
        logits_list = self.forward(obs)
        actions, log_probs = [], []
        for logits in logits_list:
            dist = Categorical(logits=logits)
            if deterministic:
                a = logits.argmax(dim=-1)
            else:
                a = dist.sample()
            actions.append(a)
            log_probs.append(dist.log_prob(a))
        action_tensor = torch.stack(actions, dim=-1)            # (batch, 3)
        total_log_prob = torch.stack(log_probs, dim=-1).sum(-1)  # (batch,)
        return action_tensor, total_log_prob


def train_reinforce(
    env,
    flatten_fn,
    num_episodes: int = 8000,
    lr: float = 1e-3,
    gamma: float = 0.99,
    print_every: int = 200,
    save_path: str = "models/reinforce_model.pt",
):
    """Train REINFORCE with baseline and return the policy + learning curve."""

    policy = ReinforcePolicy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    baseline = 0.0          # running mean of episode returns
    baseline_alpha = 0.01   # exponential moving average factor
    reward_history = []

    for ep in range(1, num_episodes + 1):
        obs, _ = env.reset()
        saved_log_probs = []
        rewards = []
        done = False

        while not done:
            obs_flat = flatten_fn(obs)
            obs_t = torch.FloatTensor(obs_flat).unsqueeze(0)

            action_t, log_prob = policy.get_action(obs_t, deterministic=False)
            action_indices = action_t.squeeze(0).numpy().astype(np.int64)

            next_obs, reward, terminated, truncated, info = env.step(action_indices)
            saved_log_probs.append(log_prob.squeeze(0))
            rewards.append(reward)
            obs = next_obs
            done = terminated or truncated

        # Compute discounted returns
        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns)

        episode_return = returns[0].item()
        reward_history.append(episode_return)

        # Update baseline
        baseline = baseline * (1 - baseline_alpha) + episode_return * baseline_alpha

        # Advantage = return - baseline
        advantages = returns - baseline

        # Policy gradient loss
        log_prob_tensor = torch.stack(saved_log_probs)
        loss = -(log_prob_tensor * advantages.detach()).mean()

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()

        if ep % print_every == 0:
            recent = reward_history[-print_every:]
            avg_ret = np.mean(recent)
            avg_cost = -avg_ret * 100  # unscale to cost
            print(f"Episode {ep:6d} | Avg Return (last {print_every}): {avg_ret:.2f} "
                  f"| Avg Cost: {avg_cost:.0f}")

    # Save the trained model
    torch.save(policy.state_dict(), save_path)
    print(f"\nModel saved to {save_path}")
    return policy, reward_history
