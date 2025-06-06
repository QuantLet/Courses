# src/financial_rl_project/agent/q_network.py
import torch
import torch.nn as nn
import numpy as np
import pandas as pd  # For isinstance checks if state might be pandas
import logging
from typing import Tuple  # For type hinting hidden_layers

log = logging.getLogger(__name__)


class QNetwork(nn.Module):
    """Neural network for approximating Q-values."""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        seed: int,
        hidden_layers: Tuple[int, ...],
        device: torch.device,
    ):
        """
        Initializes parameters and builds the Q-network model.

        Args:
            state_size (int): Dimension of each state.
            action_size (int): Dimension of each action.
            seed (int): Random seed for reproducibility.
            hidden_layers (Tuple[int, ...]): Tuple defining the number and size of hidden layers.
                                             Example: (256, 128) for two hidden layers.
            device (torch.device): The device (CPU or CUDA) to run the model on.
        """
        super().__init__()
        self.seed = torch.manual_seed(seed)
        self.device = device
        self.hidden_layers_config = hidden_layers

        layers = []
        input_dim = state_size
        for layer_size in hidden_layers:
            layers.append(nn.Linear(input_dim, layer_size))
            layers.append(nn.ReLU())
            input_dim = layer_size
        layers.append(nn.Linear(input_dim, action_size))

        self.network = nn.Sequential(*layers).to(self.device)
        log.info(
            f"QNetwork initialized with layers: input({state_size}) -> {hidden_layers} -> output({action_size}) on {self.device}"
        )

    def forward(
        self, state: torch.Tensor | np.ndarray | pd.Series | pd.DataFrame
    ) -> torch.Tensor:
        """
        Builds a network that maps state -> action values.

        Args:
            state (torch.Tensor | np.ndarray | pd.Series | pd.DataFrame): The input state.
                Can be a raw NumPy array, Pandas Series/DataFrame, or already a Tensor.
                Handled to ensure it's a FloatTensor on the correct device.

        Returns:
            torch.Tensor: The Q-values for each action from the given state.
        """
        # Convert to NumPy array if Pandas object
        if isinstance(state, (pd.Series, pd.DataFrame)):
            state = state.values

        # Convert to FloatTensor if NumPy array
        if isinstance(state, np.ndarray):
            # Ensure float32 for PyTorch compatibility
            state = torch.from_numpy(state.astype(np.float32))

        # Ensure state is a tensor and on the correct device
        if not isinstance(state, torch.Tensor):
            log.error(
                f"QNetwork input state is not a Tensor or convertible, but {type(state)}"
            )
            raise TypeError(
                f"Input state must be a Tensor or convertible, got {type(state)}"
            )

        if state.device != self.device:
            state = state.to(self.device)

        if state.dtype != torch.float32:
            state = state.float()

        # Add batch dimension if the input state is unbatched (e.g., for a single prediction)
        # Assumes states from ReplayBuffer are already batched.
        # This check is more for direct calls like agent.choose_action(single_state).
        if state.dim() == 1:
            state = state.unsqueeze(0)

        return self.network(state)
