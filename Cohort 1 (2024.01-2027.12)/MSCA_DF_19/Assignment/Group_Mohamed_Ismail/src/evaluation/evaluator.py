# src/evaluation/evaluator.py
import numpy as np
import logging
from tqdm import tqdm
from typing import (
    Callable,
    Any,
    Dict,
    Tuple,
    List,
    TYPE_CHECKING,
    Optional,
)  # Add Optional

if TYPE_CHECKING:
    from environment.financial_env import FinancialRecEnv

log = logging.getLogger(__name__)


def evaluate_policy(
    policy_callable: Callable[..., int],
    env: "FinancialRecEnv",
    num_episodes: int,
    policy_is_agent_instance: bool,
    policy_name: str,
    log_detailed_episodes: int = 0,  # <<< NEW: Number of episodes to log detailed step data for
    **kwargs_for_policy: Any,
) -> Tuple[
    float, float, float, List[float], List[int], List[List[Dict[str, Any]]]
]:  # <<< MODIFIED return
    """
    Evaluates a given policy.

    Args:
        ... (other args as before) ...
        log_detailed_episodes: If > 0, logs step-by-step data for the first N episodes.

    Returns:
        Tuple: (avg_reward, std_reward, avg_length, all_rewards_list, all_lengths_list, detailed_episode_logs)
               detailed_episode_logs is a list of lists of dictionaries.
               Outer list: episodes. Inner list: steps. Dictionary: step data.
    """
    log.info(
        f"--- Starting Evaluation for Policy: {policy_name} ({num_episodes} episodes) ---"
    )
    all_rewards: List[float] = []
    all_lengths: List[int] = []
    detailed_episode_logs: List[List[Dict[str, Any]]] = []  # <<< NEW

    for episode_num in tqdm(
        range(num_episodes), desc=f"Evaluating {policy_name}", leave=False
    ):
        state_numeric, info = env.reset()
        user_id = info.get("user_id")

        current_episode_detailed_log: List[
            Dict[str, Any]
        ] = []  # <<< NEW: For current episode's steps

        cumulative_reward = 0.0
        step_count = 0
        terminated = False
        truncated = False
        episode_specific_cache: Dict[str, Any] = {}

        # Log initial state for detailed episodes
        if episode_num < log_detailed_episodes:
            initial_portfolio_details = env.get_customer_portfolio(
                env.current_customer_id, env.current_time
            )  # type: ignore
            initial_step_data = {
                "episode": episode_num,
                "step": step_count,  # Initial state is step 0
                "customer_id": env.current_customer_id,
                "time": env.current_time.strftime("%Y-%m-%d"),  # type: ignore
                "action": None,  # No action taken yet
                "reward_for_action": None,
                "portfolio_value": sum(initial_portfolio_details.values()),
                "portfolio_hhi": env._calculate_hhi(initial_portfolio_details),  # type: ignore
                "portfolio_composition": initial_portfolio_details,
                "is_terminal": False,
            }
            current_episode_detailed_log.append(initial_step_data)

        while not (terminated or truncated):
            action_index: int
            # ... (action selection logic as before) ...
            if policy_is_agent_instance:
                current_epsilon = kwargs_for_policy.get("epsilon", 0.0)
                action_index = policy_callable(state_numeric, epsilon=current_epsilon)
            else:
                current_call_kwargs = {
                    "user_id": user_id,
                    "env_instance": env,
                    "episode_cache": episode_specific_cache,
                    **kwargs_for_policy,
                }
                action_index = policy_callable(state_numeric, **current_call_kwargs)

            next_state_numeric, reward, terminated, truncated, step_info = env.step(
                action_index
            )

            # Log step data for detailed episodes
            if episode_num < log_detailed_episodes:
                step_data = {
                    "episode": episode_num,
                    "step": step_count + 1,  # Action leads to results at end of step
                    "customer_id": step_info.get("customer_id"),
                    "time": step_info.get("current_time"),
                    "action_isin": step_info.get("action_isin"),
                    "action_index": action_index,
                    "recommendation_accepted": step_info.get("recommendation_accepted"),
                    "reward_for_action": reward,  # Reward for this step's action
                    "portfolio_value": step_info.get("portfolio_value"),
                    "portfolio_hhi": step_info.get("portfolio_hhi"),
                    "portfolio_composition": step_info.get("portfolio_composition"),
                    "is_terminal": (terminated or truncated),
                }
                current_episode_detailed_log.append(step_data)

            state_numeric = next_state_numeric
            cumulative_reward += reward
            step_count += 1

        all_rewards.append(cumulative_reward)
        all_lengths.append(step_count)
        if episode_num < log_detailed_episodes:
            detailed_episode_logs.append(
                current_episode_detailed_log
            )  # <<< ADD to main log

    avg_reward = float(np.mean(all_rewards)) if all_rewards else 0.0
    std_reward = float(np.std(all_rewards)) if all_rewards else 0.0
    avg_length = float(np.mean(all_lengths)) if all_lengths else 0.0

    log.info(f"--- Evaluation Finished for Policy: {policy_name} ---")
    # ... (logging avg reward/length) ...
    return (
        avg_reward,
        std_reward,
        avg_length,
        all_rewards,
        all_lengths,
        detailed_episode_logs,
    )
