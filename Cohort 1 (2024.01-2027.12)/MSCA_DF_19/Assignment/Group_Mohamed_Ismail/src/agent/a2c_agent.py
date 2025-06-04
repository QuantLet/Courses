# src/agent/a2c_agent.py
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical  # For sampling actions from policy
import numpy as np
import os
from typing import Optional
import logging
from omegaconf import DictConfig
from typing import List, Tuple, Any

from .actor_network import ActorNetwork
from .critic_network import CriticNetwork

log = logging.getLogger(__name__)


class A2CAgent:
    """Advantage Actor-Critic (A2C) Agent."""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        agent_cfg: DictConfig,
        global_seed: int,
        device: torch.device,
    ):
        """
        Initializes the A2C Agent.

        Args:
            state_size: Dimension of the state space.
            action_size: Number of discrete actions.
            agent_cfg: Configuration for the agent (from Hydra cfg.agent).
                       Expected: actor_hidden_layers, actor_lr,
                                 critic_hidden_layers, critic_lr,
                                 gamma, entropy_coefficient.
            global_seed: Random seed for reproducibility.
            device: PyTorch device (cpu or cuda).
        """
        self.state_size = state_size
        self.action_size = action_size
        self.agent_cfg = agent_cfg
        self.device = device
        self.seed = global_seed
        # A2C usually updates with on-policy data, so large replay buffer is not typical.
        # It collects trajectories.
        self.trajectory_states: List[torch.Tensor] = []
        self.trajectory_actions: List[torch.Tensor] = []
        self.trajectory_rewards: List[float] = []
        self.trajectory_log_probs: List[torch.Tensor] = []
        self.trajectory_values: List[
            torch.Tensor
        ] = []  # Critic's V(s) for states in trajectory
        self.trajectory_dones: List[bool] = []

        # Actor Network
        self.actor = ActorNetwork(
            state_size,
            action_size,
            self.seed,
            tuple(agent_cfg.actor_hidden_layers),
            self.device,
        )
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=agent_cfg.actor_lr
        )

        # Critic Network
        self.critic = CriticNetwork(
            state_size,
            self.seed,  # Critic outputs a single value, action_size not needed
            tuple(agent_cfg.critic_hidden_layers),
            self.device,
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=agent_cfg.critic_lr
        )

        log.info(f"A2CAgent initialized with seed {self.seed} on {self.device}.")
        log.info(
            f"  A2C Config: Gamma={agent_cfg.gamma}, EntropyCoeff={agent_cfg.entropy_coefficient}"
        )

    def choose_action(
        self, state: np.ndarray, store_for_trajectory: bool = True
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Chooses an action based on the current policy (actor network).
        Also returns log probability of the action and state value.

        Args:
            state: Current state from the environment (NumPy array).
            store_for_trajectory: If True, stores state, action, log_prob, value for later learning.

        Returns:
            Tuple of (action_index, action_log_prob, state_value_tensor)
        """
        state_tensor = (
            torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(self.device)
        )

        self.actor.eval()  # Set actor to eval mode for action selection
        self.critic.eval()  # Set critic to eval mode for value estimation
        with torch.no_grad():
            action_logits = self.actor(state_tensor)
            state_value = self.critic(state_tensor)  # V(s)
        self.actor.train()  # Back to train mode
        self.critic.train()

        action_probs = F.softmax(action_logits, dim=-1)
        distribution = Categorical(action_probs)
        action = distribution.sample()  # Sample an action
        action_log_prob = distribution.log_prob(action)  # Log prob of the chosen action

        if store_for_trajectory:
            # Store tensors directly to avoid repeated conversions
            self.trajectory_states.append(state_tensor.squeeze(0))  # Remove batch dim
            self.trajectory_actions.append(action)  # Store as tensor
            self.trajectory_log_probs.append(action_log_prob)
            self.trajectory_values.append(state_value.squeeze(-1))  # Store V(s) tensor

        return action.item(), action_log_prob, state_value  # Return action index

    def record_reward_done(self, reward: float, done: bool):
        """Records reward and done flag for the last step taken."""
        if not self.trajectory_actions:  # Should not happen if choose_action was called
            log.warning(
                "Attempted to record reward/done without a prior action in trajectory."
            )
            return
        self.trajectory_rewards.append(reward)
        self.trajectory_dones.append(done)

    def learn_from_trajectory(
        self, next_state_value_tensor: Optional[torch.Tensor] = None
    ):
        if not self.trajectory_actions:
            log.debug("No trajectory data to learn from.")
            return None, None

        # --- Prepare Tensors from Trajectory Data ---
        # States were stored as tensors
        all_states_in_trajectory_t = torch.stack(self.trajectory_states).to(self.device)
        # Actions were stored as tensors
        actions_taken_t = (
            torch.cat(self.trajectory_actions).unsqueeze(1).to(self.device)
        )
        # Log probs were stored as tensors
        log_probs_of_actions_t = (
            torch.cat(self.trajectory_log_probs).unsqueeze(1).to(self.device)
        )
        # Stored values (from choose_action) are detached, good for advantage calculation
        values_at_sampling_time_t = (
            torch.cat(self.trajectory_values).unsqueeze(1).to(self.device)
        )

        rewards_t = (
            torch.tensor(self.trajectory_rewards, dtype=torch.float32)
            .to(self.device)
            .unsqueeze(1)
        )
        # Dones need to be float for multiplication with gamma later if not careful, but bool is fine for masking
        dones_np = np.array(
            self.trajectory_dones, dtype=bool
        )  # For numpy indexing/masking if needed

        # --- Calculate Returns (Targets for Critic) ---
        returns = []
        discounted_return_accumulator = 0.0

        if next_state_value_tensor is not None and not self.trajectory_dones[-1]:
            # Bootstrap from V(s_N+1) if trajectory didn't end with a true done state
            discounted_return_accumulator = next_state_value_tensor.item()

        # Iterate backwards through the trajectory to calculate discounted returns
        for i in reversed(range(len(self.trajectory_rewards))):
            reward_step = self.trajectory_rewards[i]
            is_done_step = self.trajectory_dones[i]

            if is_done_step:
                discounted_return_accumulator = 0.0  # Value of terminal state is 0

            discounted_return_accumulator = reward_step + (
                self.agent_cfg.gamma * discounted_return_accumulator
            )
            returns.insert(0, discounted_return_accumulator)

        returns_t = (
            torch.tensor(returns, dtype=torch.float32).to(self.device).unsqueeze(1)
        )

        # --- Critic Update ---
        # Re-evaluate the critic on the trajectory states to get current V(s) WITH gradients
        self.critic.train()  # Ensure critic is in training mode
        current_values_from_critic_t = self.critic(
            all_states_in_trajectory_t
        )  # THIS IS THE KEY CHANGE

        critic_loss = F.mse_loss(current_values_from_critic_t, returns_t)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()  # This should work now
        # if self.agent_cfg.get('max_grad_norm'): # Check if max_grad_norm is configured
        #    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.agent_cfg.max_grad_norm)
        self.critic_optimizer.step()

        # --- Actor Update ---
        # Advantages: A(s_t, a_t) = Returns_t - V_at_sampling_time(s_t)
        # Use values_at_sampling_time_t here, detached, as per standard A2C.
        # The advantages should reflect the value estimate at the time of action, not the updated one.
        advantages_t = (
            returns_t - values_at_sampling_time_t
        )  # values_at_sampling_time_t is already detached implicitly
        # because it was computed in no_grad during choose_action
        # If not, .detach() would be needed: returns_t - values_at_sampling_time_t.detach()

        # Actor loss based on log probabilities of actions taken and advantages
        actor_policy_loss = (-log_probs_of_actions_t * advantages_t).mean()

        # Entropy Bonus
        self.actor.train()  # Ensure actor is in training mode
        current_action_logits = self.actor(
            all_states_in_trajectory_t
        )  # Get current policy's logits
        current_action_probs = F.softmax(current_action_logits, dim=-1)
        distribution = Categorical(probs=current_action_probs)
        entropy_bonus = distribution.entropy().mean()

        actor_loss_final = (
            actor_policy_loss - self.agent_cfg.entropy_coefficient * entropy_bonus
        )

        self.actor_optimizer.zero_grad()
        actor_loss_final.backward()
        # if self.agent_cfg.get('max_grad_norm'):
        #    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.agent_cfg.max_grad_norm)
        self.actor_optimizer.step()

        self._clear_trajectory()

        return actor_loss_final.item(), critic_loss.item()

    def _clear_trajectory(self):
        self.trajectory_states.clear()
        self.trajectory_actions.clear()
        self.trajectory_rewards.clear()
        self.trajectory_log_probs.clear()
        self.trajectory_values.clear()
        self.trajectory_dones.clear()

    def save_model(self, filepath_prefix: str):
        """Saves actor and critic network weights."""
        try:
            os.makedirs(
                os.path.dirname(filepath_prefix), exist_ok=True
            )  # Ensure dir exists
            torch.save(self.actor.state_dict(), f"{filepath_prefix}_actor.pth")
            torch.save(self.critic.state_dict(), f"{filepath_prefix}_critic.pth")
            log.info(
                f"A2C Agent actor/critic models saved with prefix {filepath_prefix}"
            )
        except Exception as e:
            log.error(f"Error saving A2C agent models: {e}", exc_info=True)

    def load_model(self, filepath_prefix: str) -> bool:
        """Loads actor and critic network weights."""
        actor_path = f"{filepath_prefix}_actor.pth"
        critic_path = f"{filepath_prefix}_critic.pth"
        loaded_actor, loaded_critic = False, False
        try:
            if os.path.exists(actor_path):
                self.actor.load_state_dict(
                    torch.load(actor_path, map_location=self.device)
                )
                self.actor.eval()
                loaded_actor = True
                log.info(f"A2C Actor model loaded from {actor_path}")
            else:
                log.warning(f"A2C Actor model file not found at {actor_path}")

            if os.path.exists(critic_path):
                self.critic.load_state_dict(
                    torch.load(critic_path, map_location=self.device)
                )
                self.critic.eval()
                loaded_critic = True
                log.info(f"A2C Critic model loaded from {critic_path}")
            else:
                log.warning(f"A2C Critic model file not found at {critic_path}")

            return loaded_actor and loaded_critic  # Return True only if both loaded
        except Exception as e:
            log.error(f"Error loading A2C agent models: {e}", exc_info=True)
            return False
