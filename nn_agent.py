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


def _obs_tensor(obs, flatten_fn, device):
    # Convert one observation to a 1 x obs_dim tensor on device.
    return torch.as_tensor(
        flatten_fn(obs), dtype=torch.float32, device=device
    ).unsqueeze(0)




def _diag_summary(losses, td_errors, selected_q, targets, grad_norms, all_q, epsilon):
    """Summarize detached tensors with only end-of-episode device syncs."""
    def cat(values):
        if not values:
            return None
        return torch.cat([v.detach().reshape(-1) for v in values])

    loss_t = cat(losses)
    td_t = cat(td_errors)
    selected_t = cat(selected_q)
    target_t = cat(targets)
    grad_t = cat(grad_norms)
    q_t = cat(all_q)

    def mean(t):
        return float(t.mean().item()) if t is not None and t.numel() else float("nan")

    def std(t):
        return float(t.std(unbiased=False).item()) if t is not None and t.numel() else float("nan")

    def abs_mean(t):
        return float(t.abs().mean().item()) if t is not None and t.numel() else float("nan")

    def abs_max(t):
        return float(t.abs().max().item()) if t is not None and t.numel() else float("nan")

    return {
        "epsilon": float(epsilon),
        "loss_mean": mean(loss_t),
        "td_error_abs_mean": abs_mean(td_t),
        "td_error_abs_max": abs_max(td_t),
        "selected_q_mean": mean(selected_t),
        "selected_q_std": std(selected_t),
        "target_mean": mean(target_t),
        "target_std": std(target_t),
        "grad_norm_mean": mean(grad_t),
        "grad_norm_max": float(grad_t.max().item())
            if grad_t is not None and grad_t.numel() else float("nan"),
        "q_min": float(q_t.min().item())
            if q_t is not None and q_t.numel() else float("nan"),
        "q_max": float(q_t.max().item())
            if q_t is not None and q_t.numel() else float("nan"),
        "q_abs_max": abs_max(q_t),
    }

def train_nn_qlearning_episode(env, model, optimizer, flatten_fn,
                                gamma=0.99, epsilon=0.1,
                                device=torch.device("cpu"),
                                return_diagnostics=False):
    """Run one online NN Q-Learning episode with optional diagnostics."""
    obs, _ = env.reset()
    done = False
    ep_cost = 0.0

    losses, td_errors = [], []
    selected_q_values, target_values = [], []
    grad_norms, all_q_values = [], []

    while not done:
        obs_t = _obs_tensor(obs, flatten_fn, device)

        with torch.no_grad():
            actions, _ = model.get_actions(obs_t, epsilon)

        action_array = np.array(actions, dtype=np.int64)
        next_obs, reward, terminated, truncated, info = env.step(action_array)
        done = terminated or truncated

        # Compute target: r + γ max_a' Q(s', a') for each product
        next_obs_t = _obs_tensor(next_obs, flatten_fn, device)
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
        step_selected, step_targets = [], []
        for i in range(3):
            q_sa = current_q[i][0, actions[i]]
            target_scalar = targets[i].squeeze()
            loss += F.mse_loss(q_sa, target_scalar)
            if return_diagnostics:
                q_det = q_sa.detach().reshape(1)
                t_det = target_scalar.detach().reshape(1)
                step_selected.append(q_det)
                step_targets.append(t_det)
                td_errors.append(t_det - q_det)

        optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if return_diagnostics:
            losses.append(loss.detach().reshape(1))
            selected_q_values.extend(step_selected)
            target_values.extend(step_targets)
            grad_norms.append(grad_norm.detach().reshape(1))
            all_q_values.append(torch.cat([
                head.detach().reshape(-1) for head in current_q
            ]))

        obs = next_obs
        ep_cost += (-reward * 100)

    if return_diagnostics:
        return ep_cost, _diag_summary(
            losses, td_errors, selected_q_values, target_values,
            grad_norms, all_q_values, epsilon)
    return ep_cost


def train_nn_sarsa_episode(env, model, optimizer, flatten_fn,
                            gamma=0.99, epsilon=0.1,
                            device=torch.device("cpu"),
                            return_diagnostics=False):
    """Run one online NN SARSA episode with optional diagnostics."""
    obs, _ = env.reset()
    obs_t = _obs_tensor(obs, flatten_fn, device)

    with torch.no_grad():
        actions, _ = model.get_actions(obs_t, epsilon)

    done = False
    ep_cost = 0.0

    losses, td_errors = [], []
    selected_q_values, target_values = [], []
    grad_norms, all_q_values = [], []

    while not done:
        action_array = np.array(actions, dtype=np.int64)
        next_obs, reward, terminated, truncated, info = env.step(action_array)
        done = terminated or truncated

        next_obs_t = _obs_tensor(next_obs, flatten_fn, device)
        current_q = model(obs_t)

        with torch.no_grad():
            next_actions, _ = model.get_actions(next_obs_t, epsilon)
            next_q_vals = model(next_obs_t)

        # SARSA target: r + γ Q(s', a') where a' is the next action taken
        loss = 0
        step_selected, step_targets = [], []
        for i in range(3):
            next_q = next_q_vals[i][0, next_actions[i]] * (1 - int(done))
            target = reward + gamma * next_q
            q_sa = current_q[i][0, actions[i]]
            loss += F.mse_loss(q_sa, target.detach())
            if return_diagnostics:
                q_det = q_sa.detach().reshape(1)
                t_det = target.detach().reshape(1)
                step_selected.append(q_det)
                step_targets.append(t_det)
                td_errors.append(t_det - q_det)

        optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if return_diagnostics:
            losses.append(loss.detach().reshape(1))
            selected_q_values.extend(step_selected)
            target_values.extend(step_targets)
            grad_norms.append(grad_norm.detach().reshape(1))
            all_q_values.append(torch.cat([
                head.detach().reshape(-1) for head in current_q
            ]))

        obs_t = next_obs_t
        actions = next_actions
        ep_cost += (-reward * 100)

    if return_diagnostics:
        return ep_cost, _diag_summary(
            losses, td_errors, selected_q_values, target_values,
            grad_norms, all_q_values, epsilon)
    return ep_cost
