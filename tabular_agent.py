"""Tabular RL agents: Q-Learning, SARSA, TD(λ) with Eligibility Traces.

The 38-dim continuous state is discretized into bins. Q-tables are factored
(one per product, 11 actions each) to keep the table size tractable.
"""

import numpy as np


class StateDiscretizer:
    """Bin continuous observations into a single integer state index."""

    def __init__(self):
        # Inventory per product: 5 bins
        self.inv_edges = np.array([30, 60, 100, 150])   # → 5 bins each
        self.n_inv = len(self.inv_edges) + 1              # 5

        # Capacity utilisation: 4 bins
        self.cap_edges = np.array([0.3, 0.6, 0.85])
        self.n_cap = len(self.cap_edges) + 1              # 4

        # Day in episode: 4 bins
        self.day_edges = np.array([12, 25, 38])
        self.n_day = len(self.day_edges) + 1              # 4

        # Pipeline load per product (sum of arrival pipeline): 3 bins
        self.pipe_edges = np.array([30, 80])
        self.n_pipe = len(self.pipe_edges) + 1            # 3

        # state size: 5³ × 4 × 4 × 3³ = 125 × 16 × 27 = 54,000
        self.n_states = (self.n_inv ** 3) * self.n_cap * self.n_day * (self.n_pipe ** 3)

    def discretize(self, obs):
        inv = np.asarray(obs["inventory"], dtype=np.float64)
        cap = np.asarray(obs["capacity_utilisation"], dtype=np.float64).item()
        day = np.asarray(obs["day"], dtype=np.float64).item()
        pipeline = np.asarray(obs["arrival_pipeline"], dtype=np.float64)
        pipe_sums = pipeline.sum(axis=1)  # per-product pipeline load

        inv_bins = [int(np.digitize(inv[i], self.inv_edges)) for i in range(3)]
        cap_bin = int(np.digitize(cap, self.cap_edges))
        day_bin = int(np.digitize(day, self.day_edges))
        pipe_bins = [int(np.digitize(pipe_sums[i], self.pipe_edges)) for i in range(3)]

        idx = inv_bins[0]
        idx = idx * self.n_inv + inv_bins[1]
        idx = idx * self.n_inv + inv_bins[2]
        idx = idx * self.n_cap + cap_bin
        idx = idx * self.n_day + day_bin
        idx = idx * self.n_pipe + pipe_bins[0]
        idx = idx * self.n_pipe + pipe_bins[1]
        idx = idx * self.n_pipe + pipe_bins[2]
        return idx

    def config_dict(self):
        """Serializable config for embedding in policy files."""
        return {
            "inv_edges": self.inv_edges.tolist(),
            "cap_edges": self.cap_edges.tolist(),
            "day_edges": self.day_edges.tolist(),
            "pipe_edges": self.pipe_edges.tolist(),
        }


class TabularAgent:
    """Factored Q-tables: one (n_states × 11) table per product."""

    def __init__(self, n_states, n_actions=11, n_products=3):
        self.n_states = n_states
        self.n_actions = n_actions
        self.n_products = n_products
        self.tables = [np.zeros((n_states, n_actions)) for _ in range(n_products)]

    def get_actions(self, state, epsilon=0.0):
        """Epsilon-greedy action selection. Returns list of 3 action indices."""
        actions = []
        for i in range(self.n_products):
            if np.random.random() < epsilon:
                actions.append(np.random.randint(self.n_actions))
            else:
                actions.append(int(np.argmax(self.tables[i][state])))
        return actions

    def get_greedy_actions(self, state):
        return [int(np.argmax(self.tables[i][state])) for i in range(self.n_products)]

    def save(self, path):
        np.savez(str(path),
                 q0=self.tables[0], q1=self.tables[1], q2=self.tables[2])

    def load(self, path):
        data = np.load(str(path))
        self.tables = [data["q0"], data["q1"], data["q2"]]




def _summary(values, *, absolute=False):
    vals = [float(v) for v in values]
    if not vals:
        return float("nan"), float("nan")
    if absolute:
        vals = [abs(v) for v in vals]
    return float(np.mean(vals)), float(np.max(vals))


def _tabular_diag(epsilon, td_errors, updates, touched_q, visited_states, n_states):
    td_mean, td_max = _summary(td_errors, absolute=True)
    upd_mean, upd_max = _summary(updates, absolute=True)
    return {
        "epsilon": float(epsilon),
        "td_error_abs_mean": td_mean,
        "td_error_abs_max": td_max,
        "q_update_abs_mean": upd_mean,
        "q_update_abs_max": upd_max,
        "q_touched_min": float(min(touched_q, default=float("nan"))),
        "q_touched_max": float(max(touched_q, default=float("nan"))),
        "q_abs_max": float(max((abs(v) for v in touched_q), default=float("nan"))),
        "state_coverage": float(len(visited_states) / float(n_states)),
    }

def train_tabular_qlearning(env, discretizer, agent, num_episodes,
                             alpha=0.1, gamma=0.99,
                             epsilon_start=1.0, epsilon_end=0.05,
                             epsilon_decay_episodes=None,
                             episode_offset=0,
                             return_diagnostics=False,
                             epsilon_schedule=None):
    """Q-Learning: off-policy TD(0) with max over next actions."""
    if epsilon_decay_episodes is None:
        epsilon_decay_episodes = int(
            (episode_offset + num_episodes) * 0.7)

    history = []
    visited_states = set()
    for ep in range(num_episodes):
        absolute_ep = episode_offset + ep
        epsilon = (
            float(epsilon_schedule(absolute_ep))
            if epsilon_schedule is not None
            else max(
                epsilon_end,
                epsilon_start
                - (epsilon_start - epsilon_end)
                * absolute_ep / epsilon_decay_episodes)
        )
        obs, _ = env.reset()
        state = discretizer.discretize(obs)
        if return_diagnostics:
            visited_states.add(state)
        done = False
        ep_cost = 0
        td_errors, updates, touched_q = [], [], []

        while not done:
            actions = agent.get_actions(state, epsilon)
            action_array = np.array(actions, dtype=np.int64)
            next_obs, reward, terminated, truncated, info = env.step(action_array)
            next_state = discretizer.discretize(next_obs)
            if return_diagnostics:
                visited_states.add(next_state)
            done = terminated or truncated

            # Q-Learning update (per product)
            for i in range(3):
                best_next = np.max(agent.tables[i][next_state])
                td_target = reward + gamma * best_next * (1 - int(done))
                old_q = agent.tables[i][state, actions[i]]
                td_error = td_target - old_q
                update = alpha * td_error
                agent.tables[i][state, actions[i]] += update
                if return_diagnostics:
                    td_errors.append(float(td_error))
                    updates.append(float(update))
                    touched_q.append(float(agent.tables[i][state, actions[i]]))

            state = next_state
            ep_cost += (-reward * 100)

        history.append(ep_cost)
        if return_diagnostics:
            yield ep, ep_cost, _tabular_diag(
                epsilon, td_errors, updates, touched_q,
                visited_states, agent.n_states)
        else:
            yield ep, ep_cost  # generator for checkpointing in train.py


def train_tabular_sarsa(env, discretizer, agent, num_episodes,
                         alpha=0.1, gamma=0.99,
                         epsilon_start=1.0, epsilon_end=0.05,
                         epsilon_decay_episodes=None,
                         episode_offset=0,
                         return_diagnostics=False,
                         epsilon_schedule=None):
    """SARSA: on-policy TD(0)."""
    if epsilon_decay_episodes is None:
        epsilon_decay_episodes = int(
            (episode_offset + num_episodes) * 0.7)

    visited_states = set()
    for ep in range(num_episodes):
        absolute_ep = episode_offset + ep
        epsilon = (
            float(epsilon_schedule(absolute_ep))
            if epsilon_schedule is not None
            else max(
                epsilon_end,
                epsilon_start
                - (epsilon_start - epsilon_end)
                * absolute_ep / epsilon_decay_episodes)
        )
        obs, _ = env.reset()
        state = discretizer.discretize(obs)
        if return_diagnostics:
            visited_states.add(state)
        actions = agent.get_actions(state, epsilon)
        done = False
        ep_cost = 0
        td_errors, updates, touched_q = [], [], []

        while not done:
            action_array = np.array(actions, dtype=np.int64)
            next_obs, reward, terminated, truncated, info = env.step(action_array)
            next_state = discretizer.discretize(next_obs)
            if return_diagnostics:
                visited_states.add(next_state)
            done = terminated or truncated

            next_actions = agent.get_actions(next_state, epsilon)

            # SARSA update (per product)
            for i in range(3):
                next_q = agent.tables[i][next_state, next_actions[i]] * (1 - int(done))
                td_target = reward + gamma * next_q
                old_q = agent.tables[i][state, actions[i]]
                td_error = td_target - old_q
                update = alpha * td_error
                agent.tables[i][state, actions[i]] += update
                if return_diagnostics:
                    td_errors.append(float(td_error))
                    updates.append(float(update))
                    touched_q.append(float(agent.tables[i][state, actions[i]]))

            state = next_state
            actions = next_actions
            ep_cost += (-reward * 100)

        if return_diagnostics:
            yield ep, ep_cost, _tabular_diag(
                epsilon, td_errors, updates, touched_q,
                visited_states, agent.n_states)
        else:
            yield ep, ep_cost


def train_td_lambda(env, discretizer, agent, num_episodes,
                     alpha=0.1, gamma=0.99, lambd=0.8,
                     epsilon_start=1.0, epsilon_end=0.05,
                     epsilon_decay_episodes=None,
                     episode_offset=0,
                     return_diagnostics=False,
                     epsilon_schedule=None):
    """TD(λ) with eligibility traces (SARSA-style, replacing traces)."""
    if epsilon_decay_episodes is None:
        epsilon_decay_episodes = int(
            (episode_offset + num_episodes) * 0.7)

    trace_decay = gamma * lambd

    visited_states = set()
    for ep in range(num_episodes):
        absolute_ep = episode_offset + ep
        epsilon = (
            float(epsilon_schedule(absolute_ep))
            if epsilon_schedule is not None
            else max(
                epsilon_end,
                epsilon_start
                - (epsilon_start - epsilon_end)
                * absolute_ep / epsilon_decay_episodes)
        )

        # Sparse replacing traces: only active state-action pairs are stored.
        traces = [dict() for _ in agent.tables]

        obs, _ = env.reset()
        state = discretizer.discretize(obs)
        if return_diagnostics:
            visited_states.add(state)
        actions = agent.get_actions(state, epsilon)
        done = False
        ep_cost = 0
        td_errors, updates, touched_q = [], [], []
        active_counts, trace_values = [], []

        while not done:
            action_array = np.array(actions, dtype=np.int64)
            next_obs, reward, terminated, truncated, info = env.step(action_array)
            next_state = discretizer.discretize(next_obs)
            if return_diagnostics:
                visited_states.add(next_state)
            done = terminated or truncated

            next_actions = agent.get_actions(next_state, epsilon)

            for i in range(3):
                next_q = agent.tables[i][next_state, next_actions[i]] * (1 - int(done))
                td_error = reward + gamma * next_q - agent.tables[i][state, actions[i]]

                # Decay only currently active traces.
                trace = traces[i]
                for key in trace:
                    trace[key] *= trace_decay

                # Replacing trace for the current state-action pair.
                trace[(state, actions[i])] = 1.0

                # Update only entries with non-zero eligibility.
                table = agent.tables[i]
                for (s, a), eligibility in trace.items():
                    update = alpha * td_error * eligibility
                    table[s, a] += update
                    if return_diagnostics:
                        updates.append(float(update))
                        touched_q.append(float(table[s, a]))

                if return_diagnostics:
                    td_errors.append(float(td_error))
                    active_counts.append(len(trace))
                    trace_values.extend(float(v) for v in trace.values())

            state = next_state
            actions = next_actions
            ep_cost += (-reward * 100)

        if return_diagnostics:
            diag = _tabular_diag(
                epsilon, td_errors, updates, touched_q,
                visited_states, agent.n_states)
            diag.update({
                "active_trace_mean": float(np.mean(active_counts))
                    if active_counts else float("nan"),
                "active_trace_max": float(max(active_counts, default=float("nan"))),
                "trace_abs_mean": float(np.mean(np.abs(trace_values)))
                    if trace_values else float("nan"),
                "trace_abs_max": float(max((abs(v) for v in trace_values),
                                           default=float("nan"))),
            })
            yield ep, ep_cost, diag
        else:
            yield ep, ep_cost
