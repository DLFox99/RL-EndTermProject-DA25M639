#!/usr/bin/env python3
"""Prove optional Phase-3 agent diagnostics do not change update mathematics."""

import copy
import numpy as np
import torch

from nn_agent import QNetworkFactored, train_nn_qlearning_episode, train_nn_sarsa_episode
from tabular_agent import TabularAgent, train_tabular_qlearning, train_tabular_sarsa, train_td_lambda


class TinyNNEnv:
    def __init__(self): self.t = 0
    def reset(self): self.t = 0; return np.zeros(38, dtype=np.float32), {}
    def step(self, action):
        self.t += 1
        obs = np.full(38, self.t / 10.0, dtype=np.float32)
        reward = -float(1 + self.t)
        return obs, reward, self.t >= 3, False, {}


def flat(x): return np.asarray(x, dtype=np.float32)


def compare_nn(fn):
    torch.manual_seed(7); np.random.seed(7)
    m1 = QNetworkFactored(hidden=8)
    m2 = copy.deepcopy(m1)
    o1 = torch.optim.SGD(m1.parameters(), lr=1e-3)
    o2 = torch.optim.SGD(m2.parameters(), lr=1e-3)

    np.random.seed(101)
    cost1 = fn(TinyNNEnv(), m1, o1, flat, epsilon=0.0)
    np.random.seed(101)
    cost2, diag = fn(TinyNNEnv(), m2, o2, flat, epsilon=0.0,
                     return_diagnostics=True)
    assert cost1 == cost2
    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        assert torch.allclose(p1, p2, atol=0, rtol=0)
    assert np.isfinite(diag["loss_mean"])
    assert np.isfinite(diag["grad_norm_mean"])


class TinyTabEnv:
    def __init__(self): self.t = 0
    def reset(self): self.t = 0; return 0, {}
    def step(self, action):
        self.t += 1
        return self.t % 4, -float(self.t), self.t >= 3, False, {}


class TinyDisc:
    n_states = 4
    def discretize(self, obs): return int(obs)


def compare_tab(fn, **kwargs):
    a1, a2 = TabularAgent(4), TabularAgent(4)
    common = dict(env=TinyTabEnv(), discretizer=TinyDisc(), num_episodes=2,
                  alpha=0.1, gamma=0.9, epsilon_start=0.0, epsilon_end=0.0,
                  epsilon_decay_episodes=1)
    np.random.seed(9)
    plain = list(fn(agent=a1, **common, **kwargs))
    common["env"] = TinyTabEnv()
    np.random.seed(9)
    rich = list(fn(agent=a2, return_diagnostics=True, **common, **kwargs))
    assert [x[:2] for x in rich] == plain
    for q1, q2 in zip(a1.tables, a2.tables):
        assert np.array_equal(q1, q2)
    assert np.isfinite(rich[-1][2]["td_error_abs_mean"])


def main():
    compare_nn(train_nn_qlearning_episode)
    compare_nn(train_nn_sarsa_episode)
    compare_tab(train_tabular_qlearning)
    compare_tab(train_tabular_sarsa)
    compare_tab(train_td_lambda, lambd=0.8)
    print("PASS: Phase-3 optional NN/tabular diagnostics preserve learning updates")


if __name__ == "__main__":
    main()
