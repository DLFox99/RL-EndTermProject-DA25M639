#!/usr/bin/env python3
"""Deterministic policy adapters for in-training periodic evaluation."""

from __future__ import annotations

import numpy as np

from env_wrappers import DiscreteActionWrapper, flatten_observation


def sb3_policy_fn(model, action_type="multi"):
    if action_type == "discrete":
        def fn(obs):
            flat = flatten_observation(obs)
            action, _ = model.predict(flat, deterministic=True)
            return DiscreteActionWrapper.decode_discrete(int(action))
        return fn

    def fn(obs):
        flat = flatten_observation(obs)
        action, _ = model.predict(flat, deterministic=True)
        return (np.asarray(action, dtype=np.int64) * 10).tolist()
    return fn


def nn_custom_policy_fn(model):
    import torch

    device = next(model.parameters()).device

    def fn(obs):
        obs_t = torch.as_tensor(
            flatten_observation(obs), dtype=torch.float32, device=device
        ).unsqueeze(0)
        with torch.no_grad():
            q_values = model(obs_t)
        return [int(q.argmax(dim=-1).item()) * 10 for q in q_values]

    return fn


def reinforce_policy_fn(policy):
    import torch

    device = next(policy.parameters()).device

    def fn(obs):
        obs_t = torch.as_tensor(
            flatten_observation(obs), dtype=torch.float32, device=device
        ).unsqueeze(0)
        with torch.no_grad():
            action_t, _ = policy.get_action(obs_t, deterministic=True)
        actions = action_t.squeeze(0).detach().cpu().numpy().astype(np.int64)
        return (actions * 10).tolist()

    return fn


def a3c_policy_fn(model):
    import torch

    device = next(model.parameters()).device

    def fn(obs):
        obs_t = torch.as_tensor(
            flatten_observation(obs), dtype=torch.float32, device=device
        ).unsqueeze(0)
        with torch.no_grad():
            actions, _, _, _ = model.get_action(obs_t, deterministic=True)
        return [int(a.item()) * 10 for a in actions]

    return fn


def tabular_policy_fn(agent, discretizer):
    def fn(obs):
        state = discretizer.discretize(obs)
        actions = agent.get_greedy_actions(state)
        return [int(a) * 10 for a in actions]
    return fn
