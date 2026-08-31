"""Shared environment wrappers for SB3 and custom training."""

import gymnasium as gym
from gymnasium import spaces
import numpy as np


def flatten_observation(obs: dict) -> np.ndarray:
    """Flatten dict observation to a normalised 38-dim float32 vector.

    Order: inventory(3) | pipeline(12) | demand_history(21) | day(1) | cap_util(1)
    """
    inventory = np.asarray(obs["inventory"], dtype=np.float32) / 200.0
    pipeline = np.asarray(obs["arrival_pipeline"], dtype=np.float32).flatten() / 100.0
    demand = np.asarray(obs["demand_history"], dtype=np.float32).flatten() / 100.0
    day = np.asarray(obs["day"], dtype=np.float32) / 50.0
    cap = np.asarray(obs["capacity_utilisation"], dtype=np.float32)
    return np.concatenate([inventory, pipeline, demand, day, cap])


OBS_DIM = 38


class FlattenObsWrapper(gym.ObservationWrapper):
    """Convert Dict observation to a flat Box for SB3 (PPO / A2C)."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=-1.0, high=10.0, shape=(OBS_DIM,), dtype=np.float32,
        )

    def observation(self, obs: dict) -> np.ndarray:
        return flatten_observation(obs)


class DiscreteActionWrapper(gym.ActionWrapper):
    """Convert MultiDiscrete([11,11,11]) → Discrete(1331) for DQN."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.action_space = spaces.Discrete(11 * 11 * 11)  # 1331

    def action(self, action: int) -> np.ndarray:
        a1 = action // 121
        a2 = (action % 121) // 11
        a3 = action % 11
        return np.array([a1, a2, a3], dtype=np.int64)

    @staticmethod
    def decode_discrete(action: int) -> list[int]:
        """Discrete index → [q1, q2, q3] actual order quantities."""
        a1 = action // 121
        a2 = (action % 121) // 11
        a3 = action % 11
        return [a1 * 10, a2 * 10, a3 * 10]
