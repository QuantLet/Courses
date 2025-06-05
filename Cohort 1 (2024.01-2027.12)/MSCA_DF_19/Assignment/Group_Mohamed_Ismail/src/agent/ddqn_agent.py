# src/agent/ddqn_agent.py
import numpy as np
import random
import torch
import torch.optim as optim
import torch.nn as nn
import os
import logging
from omegaconf import DictConfig
from typing import Tuple  # For experiences tuple

# Assuming QNetwork and ReplayBuffer are in the same 'agent' package
from .q_network import QNetwork
from .replay_buffer import ReplayBuffer

log = logging.getLogger(__name__)


class DDQNAgent:  # New Class Name
    """Interacts with and learns from the environment using Double DQN."""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        agent_cfg: DictConfig,
        global_seed: int,
        device: torch.device,
    ):
        """
        Initialize a DDQNAgent object.

        Args:
            state_size (int): Dimension of each state.
            action_size (int): Dimension of each action.
            agent_cfg (DictConfig): Configuration specific to the agent.
                                    Expected: lr, gamma, tau, buffer_size, batch_size,
                                              hidden_layers, update_every, target_update_freq.
            global_seed (int): Global random seed.
            device (torch.device): Device (CPU/GPU).
        """
        self.state_size = state_size
        self.action_size = action_size
        self.agent_cfg = agent_cfg
        self.device = device
        self.seed = global_seed
        random.seed(self.seed)

        # Q-Networks: Policy and Target
        self.qnetwork_policy = QNetwork(
            state_size,
            action_size,
            self.seed,
            tuple(agent_cfg.hidden_layers),
            self.device,
        )
        self.qnetwork_target = QNetwork(
            state_size,
            action_size,
            self.seed,
            tuple(agent_cfg.hidden_layers),
            self.device,
        )
        self.optimizer = optim.Adam(self.qnetwork_policy.parameters(), lr=agent_cfg.lr)

        self.qnetwork_target.load_state_dict(self.qnetwork_policy.state_dict())
        self.qnetwork_target.eval()

        self.memory = ReplayBuffer(
            agent_cfg.buffer_size, agent_cfg.batch_size, self.seed, self.device
        )
        self.t_step_update_learn = 0
        self.t_step_update_target = 0

        log.info(f"DDQNAgent initialized. Seed: {self.seed}.")
        log.info(
            f"  Agent Config: LR={agent_cfg.lr}, Gamma={agent_cfg.gamma}, Tau={agent_cfg.tau}, Batch={agent_cfg.batch_size}"
        )

    def step(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> float | None:
        self.memory.add(state, action, reward, next_state, done)
        self.t_step_update_learn = (
            self.t_step_update_learn + 1
        ) % self.agent_cfg.update_every
        if self.t_step_update_learn == 0:
            if len(self.memory) >= self.agent_cfg.batch_size:
                experiences = self.memory.sample()
                loss = self._learn(experiences, self.agent_cfg.gamma)
                return loss
        return None

    def choose_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        self.qnetwork_policy.eval()
        with torch.no_grad():
            action_values = self.qnetwork_policy(state)
        self.qnetwork_policy.train()
        if random.random() > epsilon:
            return int(np.argmax(action_values.cpu().data.numpy()))
        else:
            return random.choice(np.arange(self.action_size))

    def _learn(self, experiences: Tuple, gamma: float) -> float:
        """
        Update Q-network parameters using a batch of experiences with Double DQN logic.
        """
        states, actions, rewards, next_states, dones = experiences

        # --- Double DQN target calculation ---
        # 1. Get the greedy actions from the policy_network for the next_states
        self.qnetwork_policy.eval()  # Use policy net to select actions
        with torch.no_grad():
            best_actions_next_indices = self.qnetwork_policy(next_states).argmax(
                dim=1, keepdim=True
            )
        self.qnetwork_policy.train()  # Set back to train mode

        # 2. Get Q-values for these best_actions_next_indices from the target_network
        # Detach to prevent gradients from flowing through the target network during this step
        Q_targets_next_values = (
            self.qnetwork_target(next_states)
            .detach()
            .gather(1, best_actions_next_indices)
        )

        # Compute Q targets for current states: R + gamma * Q_target_network(S', argmax_a' Q_policy_network(S',a')) * (1-done)
        Q_targets = rewards + (gamma * Q_targets_next_values * (1 - dones))

        # Get expected Q-values from policy model for the actions actually taken
        Q_expected = self.qnetwork_policy(states).gather(1, actions)

        # Compute loss
        loss_fn = nn.MSELoss()
        loss = loss_fn(Q_expected, Q_targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Soft update target network
        self.t_step_update_target = (
            self.t_step_update_target + 1
        ) % self.agent_cfg.target_update_freq
        if self.t_step_update_target == 0:
            self._soft_update(
                self.qnetwork_policy, self.qnetwork_target, self.agent_cfg.tau
            )

        return loss.item()

    def _soft_update(
        self, policy_model: nn.Module, target_model: nn.Module, tau: float
    ) -> None:
        for target_param, policy_param in zip(
            target_model.parameters(), policy_model.parameters()
        ):
            target_param.data.copy_(
                tau * policy_param.data + (1.0 - tau) * target_param.data
            )

    def save_model(self, filepath: str) -> None:
        try:
            torch.save(self.qnetwork_policy.state_dict(), filepath)
            log.info(f"DDQNAgent policy network saved to {filepath}")
        except Exception as e:
            log.error(f"Error saving DDQNAgent model to {filepath}: {e}", exc_info=True)

    def load_model(self, filepath: str) -> bool:
        if not os.path.exists(filepath):
            log.warning(f"DDQNAgent model file not found at {filepath}.")
            return False
        try:
            state_dict = torch.load(filepath, map_location=self.device)
            self.qnetwork_policy.load_state_dict(state_dict)
            self.qnetwork_target.load_state_dict(self.qnetwork_policy.state_dict())
            self.qnetwork_policy.eval()
            self.qnetwork_target.eval()
            log.info(f"DDQNAgent policy network loaded from {filepath}")
            return True
        except Exception as e:
            log.error(
                f"Error loading DDQNAgent model from {filepath}: {e}", exc_info=True
            )
            return False
