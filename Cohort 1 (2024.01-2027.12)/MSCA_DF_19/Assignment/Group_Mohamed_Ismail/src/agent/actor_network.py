# src/agent/actor_network.py
import torch
import torch.nn as nn
import torch.nn.functional as F  # For softmax
import numpy as np
import pandas as pd
import logging
from typing import Tuple, Union

log = logging.getLogger(__name__)


class ActorNetwork(nn.Module):
    """Neural network for the Actor in an Actor-Critic setup."""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        seed: int,
        hidden_layers: Tuple[int, ...],
        device: torch.device,
    ):
        """
        Initializes parameters and builds the Actor network.

        Args:
            state_size: Dimension of each state.
            action_size: Dimension of each action (number of discrete actions).
            seed: Random seed.
            hidden_layers: Tuple defining the hidden layer sizes.
            device: PyTorch device (CPU or CUDA).
        """
        super().__init__()
        self.seed = torch.manual_seed(seed)
        self.device = device

        layers = []
        input_dim = state_size
        for layer_size in hidden_layers:
            layers.append(nn.Linear(input_dim, layer_size))
            layers.append(nn.ReLU())
            input_dim = layer_size
        # Output layer for action probabilities (logits before softmax)
        layers.append(nn.Linear(input_dim, action_size))

        self.network = nn.Sequential(*layers).to(self.device)
        log.info(
            f"ActorNetwork initialized: input({state_size}) -> {hidden_layers} -> output_logits({action_size}) on {self.device}"
        )

    def forward(
        self, state: Union[torch.Tensor, np.ndarray, pd.Series, pd.DataFrame]
    ) -> torch.Tensor:
        """
        Maps state to action logits.

        Args:
            state: The input state.

        Returns:
            torch.Tensor: Action logits. Softmax should be applied outside if probabilities are needed directly.
        """
        if isinstance(state, (pd.Series, pd.DataFrame)):
            state = state.values
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state.astype(np.float32))

        if not isinstance(state, torch.Tensor):
            log.error(
                f"ActorNetwork input state is not Tensor or convertible, but {type(state)}"
            )
            raise TypeError(
                f"Input state must be Tensor or convertible, got {type(state)}"
            )

        if state.device != self.device:
            state = state.to(self.device)
        if state.dtype != torch.float32:
            state = state.float()
        if state.dim() == 1:  # Add batch dimension if unbatched
            state = state.unsqueeze(0)

        action_logits = self.network(state)
        return action_logits  # Return logits, apply softmax in agent for action selection or loss calculation
