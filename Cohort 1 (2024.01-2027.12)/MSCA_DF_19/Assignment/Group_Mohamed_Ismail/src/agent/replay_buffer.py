# src/financial_rl_project/agent/replay_buffer.py
import random
import torch
import numpy as np
import pandas as pd  # For isinstance checks if state might be pandas
from collections import deque, namedtuple
import logging
from typing import Tuple, Any  # For Transition and sample return type

log = logging.getLogger(__name__)

# Define the structure of an experience transition
Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "done")
)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(
        self, buffer_size: int, batch_size: int, seed: int, device: torch.device
    ):
        """
        Initialize a ReplayBuffer object.

        Args:
            buffer_size (int): Maximum size of buffer.
            batch_size (int): Size of each training batch.
            seed (int): Random seed.
            device (torch.device): Device (CPU/GPU) to send tensors to.
        """
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.device = device
        # Note: random.seed() affects the global random state.
        # For more isolated randomness, one might use a np.random.Generator instance.
        # However, for consistency with original code and simplicity:
        random.seed(seed)  # This sets the seed for the `random` module.
        # If using np.random.choice or sample in other parts, ensure np.random.seed is also set.
        log.info(
            f"ReplayBuffer initialized with size {buffer_size}, batch_size {batch_size} on {self.device}."
        )

    def add(
        self, state: Any, action: Any, reward: Any, next_state: Any, done: Any
    ) -> None:
        """
        Add a new experience to memory.
        Input states can be pandas Series/DataFrame or numpy arrays.
        They are converted to float32 numpy arrays before storing.
        """
        # Ensure states are float32 numpy arrays before adding to buffer
        if isinstance(state, (pd.Series, pd.DataFrame)):
            state_np = state.values.astype(np.float32)
        elif isinstance(state, np.ndarray):
            state_np = state.astype(np.float32)
        else:  # Fallback if it's a list or other convertible type
            try:
                state_np = np.array(state, dtype=np.float32)
            except Exception as e:
                log.error(
                    f"Could not convert state to np.ndarray: {state}, type: {type(state)}. Error: {e}"
                )
                raise TypeError("State must be convertible to a float32 NumPy array.")

        if isinstance(next_state, (pd.Series, pd.DataFrame)):
            next_state_np = next_state.values.astype(np.float32)
        elif isinstance(next_state, np.ndarray):
            next_state_np = next_state.astype(np.float32)
        else:
            try:
                next_state_np = np.array(next_state, dtype=np.float32)
            except Exception as e:
                log.error(
                    f"Could not convert next_state to np.ndarray: {next_state}, type: {type(next_state)}. Error: {e}"
                )
                raise TypeError(
                    "Next state must be convertible to a float32 NumPy array."
                )

        experience = Transition(
            state_np, int(action), float(reward), next_state_np, bool(done)
        )
        self.memory.append(experience)

    def sample(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Randomly sample a batch of experiences from memory."""
        if len(self.memory) < self.batch_size:
            log.warning(
                f"Requested sample of size {self.batch_size}, but buffer only contains {len(self.memory)} samples."
            )
            # Decide behavior: raise error, return None, or sample with replacement?
            # Original code raised ValueError. Let's stick to that for now.
            raise ValueError(
                "Not enough experiences in replay buffer to sample a batch."
            )

        experiences = random.sample(self.memory, k=self.batch_size)

        # Convert batch of experiences to Torch Tensors on the specified device
        # np.vstack is robust for states that are 1D arrays.
        # Ensure states are correctly shaped (e.g., if they were multi-dimensional)
        states = (
            torch.from_numpy(np.vstack([e.state for e in experiences]))
            .float()
            .to(self.device)
        )
        actions = (
            torch.from_numpy(np.vstack([e.action for e in experiences]))
            .long()
            .to(self.device)
        )
        rewards = (
            torch.from_numpy(np.vstack([e.reward for e in experiences]))
            .float()
            .to(self.device)
        )
        next_states = (
            torch.from_numpy(np.vstack([e.next_state for e in experiences]))
            .float()
            .to(self.device)
        )
        dones = (
            torch.from_numpy(np.vstack([e.done for e in experiences]).astype(np.uint8))
            .float()
            .to(self.device)
        )

        return (states, actions, rewards, next_states, dones)

    def __len__(self) -> int:
        """Return the current size of internal memory."""
        return len(self.memory)
