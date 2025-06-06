# src/agent/critic_network.py
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import logging
from typing import Tuple, Union

log = logging.getLogger(__name__)


class CriticNetwork(nn.Module):
    """Neural network for the Critic in an Actor-Critic setup (estimates V(s))."""

    def __init__(
        self,
        state_size: int,
        seed: int,
        hidden_layers: Tuple[int, ...],
        device: torch.device,
    ):
        """
        Initializes parameters and builds the Critic network.

        Args:
            state_size: Dimension of each state.
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
        # Output layer for state value (single scalar output)
        layers.append(nn.Linear(input_dim, 1))

        self.network = nn.Sequential(*layers).to(self.device)
        log.info(
            f"CriticNetwork initialized: input({state_size}) -> {hidden_layers} -> output_value(1) on {self.device}"
        )

    def forward(
        self, state: Union[torch.Tensor, np.ndarray, pd.Series, pd.DataFrame]
    ) -> torch.Tensor:
        """
        Maps state to state-value V(s).

        Args:
            state: The input state.

        Returns:
            torch.Tensor: Estimated state value.
        """
        if isinstance(state, (pd.Series, pd.DataFrame)):
            state = state.values
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state.astype(np.float32))

        if not isinstance(state, torch.Tensor):
            log.error(
                f"CriticNetwork input state not Tensor or convertible, but {type(state)}"
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

        state_value = self.network(state)
        return state_value
