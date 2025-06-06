# src/financial_rl_project/agent/dqn_agent.py
import numpy as np
import random
import torch
from typing import Tuple
import torch.optim as optim
import torch.nn as nn  # For nn.MSELoss
import os
import logging
from omegaconf import DictConfig  # For agent_cfg type hint

# Relative imports from the same 'agent' package
from .q_network import QNetwork
from .replay_buffer import ReplayBuffer

log = logging.getLogger(__name__)


class DQNAgent:
    """Interacts with and learns from the environment using DQN."""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        agent_cfg: DictConfig,
        global_seed: int,
        device: torch.device,
    ):
        """
        Initialize a DQNAgent object.

        Args:
            state_size (int): Dimension of each state.
            action_size (int): Dimension of each action.
            agent_cfg (DictConfig): Configuration specific to the agent (e.g., cfg.agent from Hydra).
                                    Expected to contain lr, gamma, tau, buffer_size, batch_size,
                                    hidden_layers, update_every, target_update_freq.
            global_seed (int): Global random seed for reproducibility.
            device (torch.device): Device (CPU/GPU) for PyTorch models and tensors.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.agent_cfg = agent_cfg
        self.device = device

        # Use the global seed for agent's components for consistent initialization
        self.seed = global_seed
        random.seed(self.seed)  # For agent's own random choices (e.g., epsilon-greedy)
        # Note: np.random.seed should be set globally once (e.g. in common_utils.set_seeds)

        # Q-Networks
        self.qnetwork_policy = QNetwork(
            state_size,
            action_size,
            self.seed,
            tuple(agent_cfg.hidden_layers),  # Ensure it's a tuple
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

        # Initialize target network to match policy network
        self.qnetwork_target.load_state_dict(self.qnetwork_policy.state_dict())
        self.qnetwork_target.eval()  # Target network is not trained directly

        # Replay memory
        self.memory = ReplayBuffer(
            agent_cfg.buffer_size, agent_cfg.batch_size, self.seed, self.device
        )

        # Counters for learning and target network updates
        self.t_step_update_learn = 0  # For self.agent_cfg.update_every
        self.t_step_update_target = 0  # For self.agent_cfg.target_update_freq

        log.info(
            f"DQNAgent initialized with seed {self.seed}. Policy/Target networks on {self.device}."
        )
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
        """
        Save experience in replay memory, and learn if it's time.

        Args:
            state: Current state from the environment.
            action: Action taken.
            reward: Reward received.
            next_state: Next state received.
            done: Boolean indicating if the episode has finished.

        Returns:
            float | None: The loss value if learning occurred, otherwise None.
        """
        self.memory.add(state, action, reward, next_state, done)

        # Learn every `update_every` time steps.
        self.t_step_update_learn = (
            self.t_step_update_learn + 1
        ) % self.agent_cfg.update_every
        if self.t_step_update_learn == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) >= self.agent_cfg.batch_size:
                experiences = self.memory.sample()
                loss = self._learn(experiences, self.agent_cfg.gamma)
                return loss
        return None

    def choose_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """
        Returns an action for the given state as per current policy (epsilon-greedy).

        Args:
            state (np.ndarray): Current state from the environment.
            epsilon (float): Epsilon for epsilon-greedy action selection.

        Returns:
            int: The chosen action index.
        """
        # QNetwork's forward method handles numpy to tensor conversion, device placement, and batching
        # state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device) # Old way

        self.qnetwork_policy.eval()  # Set policy network to evaluation mode for action selection
        with torch.no_grad():
            # QNetwork.forward expects a single unbatched state or a batched state.
            # Here, we pass a single unbatched state. It will add batch dim internally.
            action_values = self.qnetwork_policy(state)
        self.qnetwork_policy.train()  # Set policy network back to training mode

        # Epsilon-greedy action selection
        if random.random() > epsilon:
            # .cpu().data.numpy() is robust for getting numpy array from tensor
            return int(np.argmax(action_values.cpu().data.numpy()))
        else:
            return random.choice(np.arange(self.action_size))

    def _learn(self, experiences: Tuple, gamma: float) -> float:
        """
        Update Q-network parameters using a batch of experiences (DQN).
        (Standard DQN, not DDQN yet based on original train.py's agent)
        """
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q-values (for next states) from target model
        # .detach() is used because we don't want to backpropagate gradients through the target network
        Q_targets_next = (
            self.qnetwork_target(next_states).detach().max(dim=1)[0].unsqueeze(1)
        )

        # Compute Q targets for current states: R + gamma * max_a' Q_target(s', a') * (1-done)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q-values from policy model for the actions taken
        # .gather(1, actions) selects the Q-value corresponding to the action taken in each state
        Q_expected = self.qnetwork_policy(states).gather(1, actions)

        # Compute loss
        loss_fn = nn.MSELoss()
        loss = loss_fn(Q_expected, Q_targets)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        # Optional: Gradient clipping (can help stabilize training for some problems)
        # torch.nn.utils.clip_grad_norm_(self.qnetwork_policy.parameters(), 1.0)
        self.optimizer.step()

        # --- Update target network ---
        # Soft update the target network towards the policy network
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
        """
        Soft update model parameters.
        θ_target = τ*θ_policy + (1 - τ)*θ_target
        """
        for target_param, policy_param in zip(
            target_model.parameters(), policy_model.parameters()
        ):
            target_param.data.copy_(
                tau * policy_param.data + (1.0 - tau) * target_param.data
            )

    def save_model(self, filepath: str) -> None:
        """Saves the policy network weights to a file."""
        try:
            # Ensure directory exists if filepath includes directories
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            torch.save(self.qnetwork_policy.state_dict(), filepath)
            log.info(f"DQN Agent policy network saved to {filepath}")
        except Exception as e:
            log.error(f"Error saving DQN agent model to {filepath}: {e}", exc_info=True)

    def load_model(self, filepath: str) -> bool:
        """Loads the policy network weights from a file."""
        if not os.path.exists(filepath):
            log.warning(f"DQN model file not found at {filepath}. Cannot load weights.")
            return False
        try:
            # Load weights to the appropriate device
            state_dict = torch.load(filepath, map_location=self.device)
            self.qnetwork_policy.load_state_dict(state_dict)
            # Also sync the target network and set both to evaluation mode
            self.qnetwork_target.load_state_dict(self.qnetwork_policy.state_dict())
            self.qnetwork_policy.eval()
            self.qnetwork_target.eval()
            log.info(f"DQN Agent policy network loaded from {filepath}")
            return True
        except Exception as e:
            log.error(
                f"Error loading DQN agent model from {filepath}: {e}", exc_info=True
            )
            return False
