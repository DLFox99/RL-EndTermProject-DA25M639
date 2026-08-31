"""Neural Network Q-Learning and SARSA — custom PyTorch implementations.

Unlike DQN, these do NOT use experience replay or a target network.
They are online, single-sample update methods with function approximation.

Action space: factored (3 heads × 11 actions) for tractability.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetworkFactored(nn.Module):
    """MLP with 3 independent Q-value heads (one per product)."""

    def __init__(self, obs_dim=38, hidden=128, n_actions=11):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.heads = nn.ModuleList([
            nn.Linear(hidden, n_actions) for _ in range(3)
        ])

    def forward(self, x):
        """Returns list of 3 tensors, each (batch, 11)."""
        h = self.shared(x)
        return [head(h) for head in self.heads]

    def get_actions(self, obs_tensor, epsilon=0.0):
        """Epsilon-greedy. Returns (actions [3], q_values [3])."""
        q_values = self.forward(obs_tensor)
        actions = []
        for q in q_values:
            if np.random.random() < epsilon:
                a = np.random.randint(q.shape[-1])
            else:
                a = q.argmax(dim=-1).item()
            actions.append(a)
        return actions, q_values


def train_nn_qlearning_episode(env, model, optimizer, flatten_fn,
                                gamma=0.99, epsilon=0.1):
    """Run one episode of online NN Q-Learning. Returns episode cost."""
    obs, _ = env.reset()
    done = False
    ep_cost = 0.0

    while not done:
        obs_t = torch.FloatTensor(flatten_fn(obs)).unsqueeze(0)

        with torch.no_grad():
            actions, _ = model.get_actions(obs_t, epsilon)

        action_array = np.array(actions, dtype=np.int64)
        next_obs, reward, terminated, truncated, info = env.step(action_array)
        done = terminated or truncated

        # Compute target: r + γ max_a' Q(s', a') for each product
        next_obs_t = torch.FloatTensor(flatten_fn(next_obs)).unsqueeze(0)
        current_q = model(obs_t)

        with torch.no_grad():
            next_q = model(next_obs_t)
            targets = []
            for i in range(3):
                max_next_q = next_q[i].max(dim=-1).values
                target = reward + gamma * max_next_q * (1 - int(done))
                targets.append(target)

        # Loss: sum of per-product TD errors
        loss = 0
        for i in range(3):
            q_sa = current_q[i][0, actions[i]]
            loss += F.mse_loss(q_sa, targets[i].squeeze())

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        obs = next_obs
        ep_cost += (-reward * 100)

    return ep_cost


def train_nn_sarsa_episode(env, model, optimizer, flatten_fn,
                            gamma=0.99, epsilon=0.1):
    """Run one episode of online NN SARSA. Returns episode cost."""
    obs, _ = env.reset()
    obs_t = torch.FloatTensor(flatten_fn(obs)).unsqueeze(0)

    with torch.no_grad():
        actions, _ = model.get_actions(obs_t, epsilon)

    done = False
    ep_cost = 0.0

    while not done:
        action_array = np.array(actions, dtype=np.int64)
        next_obs, reward, terminated, truncated, info = env.step(action_array)
        done = terminated or truncated

        next_obs_t = torch.FloatTensor(flatten_fn(next_obs)).unsqueeze(0)
        current_q = model(obs_t)

        with torch.no_grad():
            next_actions, _ = model.get_actions(next_obs_t, epsilon)
            next_q_vals = model(next_obs_t)

        # SARSA target: r + γ Q(s', a') where a' is the next action taken
        loss = 0
        for i in range(3):
            next_q = next_q_vals[i][0, next_actions[i]] * (1 - int(done))
            target = reward + gamma * next_q
            q_sa = current_q[i][0, actions[i]]
            loss += F.mse_loss(q_sa, target.detach())

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        obs_t = next_obs_t
        actions = next_actions
        ep_cost += (-reward * 100)

    return ep_cost
