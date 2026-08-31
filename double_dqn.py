"""Double DQN: subclass of SB3 DQN with double Q-learning target computation.

Standard DQN:  uses target network for both action selection AND evaluation.
Double  DQN:   uses *online* network to SELECT the best next action,
               then the *target* network to EVALUATE that action.
This reduces overestimation bias.
"""

from typing import Optional

import numpy as np
import torch as th
import torch.nn.functional as F
from stable_baselines3 import DQN


class DoubleDQN(DQN):
    """SB3 DQN with double Q-learning update rule."""

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self._update_learning_rate(self.policy.optimizer)
        losses = []

        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(
                batch_size, env=self._vec_normalize_env
            )

            with th.no_grad():
                # --- Double DQN change ---
                # Online network selects the best next action
                next_q_online = self.q_net(replay_data.next_observations)
                next_actions = next_q_online.argmax(dim=1, keepdim=True)

                # Target network evaluates that action
                next_q_target = self.q_net_target(replay_data.next_observations)
                next_q_values = next_q_target.gather(1, next_actions).squeeze(1)
                # -------------------------

                target_q_values = (
                    replay_data.rewards.flatten()
                    + (1 - replay_data.dones.flatten()) * self.gamma * next_q_values
                )

            # Current Q-values for the taken actions
            current_q_values = self.q_net(replay_data.observations)
            current_q_values = th.gather(
                current_q_values, dim=1, index=replay_data.actions.long()
            )

            loss = F.smooth_l1_loss(current_q_values.squeeze(), target_q_values)
            losses.append(loss.item())

            self.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.max_grad_norm
            )
            self.policy.optimizer.step()

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))
