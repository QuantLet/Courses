# src/training/core_train_loops.py
import numpy as np
import logging
from typing import Tuple, TYPE_CHECKING, Optional
import torch  # For A2C next_state_value

# For type hinting without circular imports
if TYPE_CHECKING:
    from environment.financial_env import (
        FinancialRecEnv,
    )  # Assuming flat src where environment is importable
    from agent.dqn_agent import DQNAgent
    from agent.a2c_agent import A2CAgent

log = logging.getLogger(__name__)


def train_dqn_episode(
    env: "FinancialRecEnv", agent: "DQNAgent", epsilon: float
) -> Tuple[float, int, float]:
    """
    Runs a single training episode for the DQN agent.

    Args:
        env: The financial recommendation environment instance.
        agent: The DQN agent instance.
        epsilon: The current epsilon value for epsilon-greedy action selection.

    Returns:
        A tuple containing:
            - cumulative_reward (float): Total reward accumulated in the episode.
            - steps (int): Number of steps taken in the episode.
            - average_loss (float): Average training loss per agent learning step in the episode.
    """
    state_numeric, info = env.reset()
    customer_id = info.get("customer_id", "UnknownCust")
    log.debug(f"DQN Ep Start: Cust={customer_id}, Eps={epsilon:.4f}")

    cumulative_reward = 0.0
    episode_steps = 0
    episode_losses = []  # Losses from agent.step()
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action_index = agent.choose_action(state_numeric, epsilon)

        next_state_numeric, reward, terminated, truncated, step_info = env.step(
            action_index
        )

        # Agent learns from the experience
        # The DQNAgent's step method handles adding to buffer and learning if ready
        loss = agent.step(
            state_numeric,
            action_index,
            reward,
            next_state_numeric,
            (terminated or truncated),
        )
        if loss is not None:  # Loss is returned only if learning happened
            episode_losses.append(loss)

        state_numeric = next_state_numeric
        cumulative_reward += reward
        episode_steps += 1

    avg_loss_this_episode = np.mean(episode_losses) if episode_losses else 0.0
    log.debug(
        f"DQN Ep End: Cust={customer_id}, Steps={episode_steps}, Reward={cumulative_reward:.2f}, AvgLoss={avg_loss_this_episode:.4f}"
    )

    return cumulative_reward, episode_steps, avg_loss_this_episode


def train_a2c_episode(
    env: "FinancialRecEnv",
    agent: "A2CAgent",
    # update_horizon: Optional[int] = None # For n-step, not implemented in this basic version
) -> Tuple[float, int, Optional[float], Optional[float]]:
    """
    Runs a single training episode for the A2C agent.
    The A2C agent collects a full trajectory during the episode and
    learns from it at the end of the episode.

    Args:
        env: The financial recommendation environment instance.
        agent: The A2C agent instance.

    Returns:
        A tuple containing:
            - cumulative_reward (float): Total reward accumulated in the episode.
            - steps (int): Number of steps taken in the episode.
            - actor_loss (Optional[float]): Actor loss from the end-of-episode update.
            - critic_loss (Optional[float]): Critic loss from the end-of-episode update.
    """
    state_numeric, info = env.reset()
    customer_id = info.get("customer_id", "UnknownCust")
    log.debug(f"A2C Ep Start: Cust={customer_id}")

    cumulative_reward = 0.0
    episode_steps = 0
    terminated = False
    truncated = False

    agent._clear_trajectory()  # Start with a fresh trajectory for the episode

    while not (terminated or truncated):
        # choose_action for A2C also stores data in agent's trajectory lists
        action_index, _action_log_prob, _state_value = agent.choose_action(
            state_numeric, store_for_trajectory=True
        )

        next_state_numeric, reward, terminated, truncated, step_info = env.step(
            action_index
        )

        # Record reward and done status for the current step in the trajectory
        agent.record_reward_done(reward, (terminated or truncated))

        state_numeric = next_state_numeric  # Current state becomes next_state for the next iteration
        cumulative_reward += reward
        episode_steps += 1

    # Learn from the collected trajectory at the end of the episode
    next_value_for_bootstrap: Optional[torch.Tensor] = None
    if (
        not terminated and truncated
    ):  # Episode ended due to truncation (max steps), not terminal state
        # Bootstrap the value of the last next_state if the episode was truncated
        # This 'state_numeric' is the S_T (last state reached before truncation)
        # The value needed is V(S_{T+1}), which is V(next_state_numeric from the last step)
        # However, by this point, state_numeric IS the last next_state_numeric.
        with torch.no_grad():
            agent.critic.eval()  # Ensure critic is in eval mode for bootstrapping
            # The 'state_numeric' at this point is the final S_t+1 if truncated
            next_state_tensor = (
                torch.from_numpy(state_numeric.astype(np.float32))
                .unsqueeze(0)
                .to(agent.device)
            )
            next_value_for_bootstrap = agent.critic(next_state_tensor)
            agent.critic.train()
    # If 'terminated' is True, the value of the terminal state is 0, so next_value_for_bootstrap remains None
    # (which agent.learn_from_trajectory should handle as R_T = r_T + gamma * 0)

    actor_loss_item, critic_loss_item = agent.learn_from_trajectory(
        next_state_value_tensor=next_value_for_bootstrap
    )

    log.debug(
        f"A2C Ep End: Cust={customer_id}, Steps={episode_steps}, Reward={cumulative_reward:.2f}, "
        f"ActorLoss={actor_loss_item if actor_loss_item is not None else 'N/A'}, "
        f"CriticLoss={critic_loss_item if critic_loss_item is not None else 'N/A'}"
    )

    return cumulative_reward, episode_steps, actor_loss_item, critic_loss_item
