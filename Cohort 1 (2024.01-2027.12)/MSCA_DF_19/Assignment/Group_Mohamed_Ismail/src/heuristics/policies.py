# src/heuristics/policies.py
import numpy as np
import pandas as pd
import logging
from typing import Dict, Callable, Any, TYPE_CHECKING, List, Tuple
import random  # For fallback if heuristic fails
import os
from datetime import datetime
from typing import Optional

# Import utilities from the same package
from .utils import (
    normalise_weights,
    calculate_mu_sigma_hat,
    get_user_risk_budget_from_env,
    get_current_date_from_env,
)

# Import CVXPY_OK from common_utils to conditionally use MVO
from utils.common_utils import CVXPY_OK  # Assuming flat src structure

if CVXPY_OK:
    import cvxpy as cp  # type: ignore

if TYPE_CHECKING:
    from environment.financial_env import FinancialRecEnv  # Assuming flat src structure
    from omegaconf import DictConfig

log = logging.getLogger(__name__)

# --- Individual Heuristic Weight Calculation Functions ---


def mvo_weights_policy(
    mu_hat: np.ndarray,
    Sigma_hat: np.ndarray,
    sigma_cap: float,
    num_assets: int,
    cvxpy_solver: Optional[str] = None,
) -> np.ndarray:
    """Calculates Mean-Variance Optimal weights."""
    if not CVXPY_OK:
        log.warning(
            "MVO policy called but CVXPY not available. Falling back to Equal Weight."
        )
        return np.full(num_assets, 1.0 / num_assets, dtype=float)

    if (
        mu_hat.shape[0] != num_assets
        or Sigma_hat.shape[0] != num_assets
        or Sigma_hat.shape[1] != num_assets
    ):
        log.error(
            f"MVO input shape mismatch: mu({mu_hat.shape}), Sigma({Sigma_hat.shape}), num_assets({num_assets}). Fallback to EW."
        )
        return np.full(num_assets, 1.0 / num_assets, dtype=float)

    try:
        w = cp.Variable(num_assets)
        # Maximize return for a risk cap (volatility constraint)
        # Ensure Sigma_hat is positive semi-definite for cp.quad_form
        # Sigma_hat from calculate_mu_sigma_hat has a small identity added, should be PSD.
        prob = cp.Problem(
            cp.Maximize(mu_hat @ w),
            [cp.sum(w) == 1, w >= 0, cp.quad_form(w, Sigma_hat) <= sigma_cap**2],
        )
        solve_kwargs = {"verbose": False}
        if cvxpy_solver:  # If a specific solver is configured via Hydra
            solve_kwargs["solver"] = cvxpy_solver.upper()  # e.g., "SCS", "ECOS"

        prob.solve(**solve_kwargs)

        if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            log.warning(
                f"CVXPY MVO solve status: {prob.status}. Falling back to Equal Weight."
            )
            return np.full(num_assets, 1.0 / num_assets, dtype=float)
        if w.value is None:
            log.warning(
                "CVXPY MVO solver returned None weights. Falling back to Equal Weight."
            )
            return np.full(num_assets, 1.0 / num_assets, dtype=float)

        return normalise_weights(w.value)
    except cp.SolverError as se:
        log.warning(f"CVXPY SolverError in MVO: {se}. Falling back to Equal Weight.")
        return np.full(num_assets, 1.0 / num_assets, dtype=float)
    except Exception as e:
        log.error(
            f"Unexpected error in MVO calculation: {e}. Falling back to Equal Weight.",
            exc_info=True,
        )
        return np.full(num_assets, 1.0 / num_assets, dtype=float)


def risk_parity_weights_policy(
    Sigma_hat: np.ndarray,
    num_assets: int,
    # sigma_cap: float # Standard RP doesn't use sigma_cap for weight calculation
) -> np.ndarray:
    """Calculates Risk Parity weights (standard version)."""
    if Sigma_hat.shape[0] != num_assets or Sigma_hat.shape[1] != num_assets:
        log.error(
            f"RP input Sigma shape mismatch: Sigma({Sigma_hat.shape}), num_assets({num_assets}). Fallback to EW."
        )
        return np.full(num_assets, 1.0 / num_assets, dtype=float)

    asset_volatilities = np.sqrt(np.diag(Sigma_hat))

    # Handle assets with zero or near-zero variance to avoid division by zero
    inv_volatilities = np.zeros_like(asset_volatilities)
    non_zero_vol_mask = asset_volatilities > 1e-9  # Threshold for non-zero volatility

    if not np.any(non_zero_vol_mask):  # All assets have zero/tiny volatility
        log.warning(
            "Risk Parity: All assets have near-zero volatility. Falling back to Equal Weight."
        )
        return np.full(num_assets, 1.0 / num_assets, dtype=float)

    inv_volatilities[non_zero_vol_mask] = 1.0 / asset_volatilities[non_zero_vol_mask]

    raw_rp_weights = inv_volatilities
    return normalise_weights(raw_rp_weights)


def equal_weights_policy(num_assets: int) -> np.ndarray:
    """Returns Equal Weights for the given number of assets."""
    if num_assets <= 0:
        log.warning(
            "Equal Weights policy called with num_assets <= 0. Returning empty array."
        )
        return np.array([])
    return np.full(num_assets, 1.0 / num_assets, dtype=float)


# --- Main Heuristic Policy Decision Function Wrapper ---

_heuristic_episode_cache: Dict[
    str, np.ndarray
] = {}  # Simple module-level cache for weights per episode


def get_heuristic_policy_decision_func(
    strategy_name: str,
    # These are passed via **kwargs from evaluate_policy in run.py
    # prices_df: pd.DataFrame, # Will be passed via kwargs_for_policy
    # cfg_heuristics: 'DictConfig',
    # num_assets: int,
    # asset_isins_ordered: List[str] # Needed by calculate_mu_sigma_hat
) -> Callable[
    [np.ndarray, Any], int
]:  # Returns a callable: (state, **kwargs) -> action_index
    """
    Returns a decision-making function for the specified heuristic strategy.
    The returned function is compatible with evaluate_policy's policy_callable.
    It handles caching of weights per episode (based on user_id).
    """
    global _heuristic_episode_cache  # Use the module-level cache

    def _heuristic_decision_maker(state: np.ndarray, **kwargs) -> int:
        """
        Inner function that makes the decision.
        kwargs are expected to be passed by evaluate_policy and include:
        user_id, env_instance, prices_df, cfg_heuristics, num_assets, episode_cache (unused here, we use module cache)
        """
        user_id: Optional[str] = kwargs.get("user_id")
        env_instance: Optional["FinancialRecEnv"] = kwargs.get("env_instance")
        prices_df_passed: Optional[pd.DataFrame] = kwargs.get("prices_df")
        cfg_heuristics_passed: Optional["DictConfig"] = kwargs.get("cfg_heuristics")
        num_assets_passed: Optional[int] = kwargs.get("num_assets")
        # asset_isins_ordered_passed: Optional[List[str]] = kwargs.get("asset_isins_ordered") # Get from env

        if not all(
            [
                user_id,
                env_instance,
                prices_df_passed is not None,
                cfg_heuristics_passed,
                num_assets_passed is not None,
            ]
        ):
            log.error(
                "Heuristic decision maker missing critical kwargs (user_id, env_instance, prices_df, cfg_heuristics, num_assets). Falling back to random."
            )
            return random.choice(
                np.arange(num_assets_passed or 1)
            )  # Default to 1 if num_assets is 0 or None

        # Use a unique key for caching within an episode (user_id + current_episode_step or just user_id if weights are fixed per episode)
        # The original logic implies weights are fixed per episode for a user.
        cache_key = f"{user_id}_{env_instance.current_episode_step if hasattr(env_instance, 'current_episode_step') else 'ep_start'}"

        target_weights: np.ndarray
        if cache_key in _heuristic_episode_cache:
            target_weights = _heuristic_episode_cache[cache_key]
            log.debug(f"Heuristic {strategy_name} using cached weights for {cache_key}")
        else:
            log.debug(
                f"Heuristic {strategy_name} calculating new weights for {cache_key}"
            )
            current_sim_date = get_current_date_from_env(env_instance)
            lookback = cfg_heuristics_passed.lookback_days

            # Get asset ISINs in the order the environment uses them
            asset_isins_env_order = env_instance.asset_isins_list

            mu_hat, sigma_hat = calculate_mu_sigma_hat(
                prices_df_passed, asset_isins_env_order, current_sim_date, lookback
            )

            user_sigma_cap = get_user_risk_budget_from_env(env_instance, user_id)

            if strategy_name == "MVO":
                target_weights = mvo_weights_policy(
                    mu_hat,
                    sigma_hat,
                    user_sigma_cap,
                    num_assets_passed,
                    cfg_heuristics_passed.get("cvxpy_solver"),
                )
            elif strategy_name == "Risk Parity":
                target_weights = risk_parity_weights_policy(
                    sigma_hat, num_assets_passed
                )
            elif strategy_name == "Equal Weight":
                target_weights = equal_weights_policy(num_assets_passed)
            else:
                log.error(
                    f"Unknown heuristic strategy: {strategy_name}. Falling back to random."
                )
                return random.choice(np.arange(num_assets_passed))

            # Ensure weights match num_assets, fallback if calculation failed badly
            if target_weights.shape[0] != num_assets_passed:
                log.warning(
                    f"Heuristic {strategy_name} weight dimension mismatch ({target_weights.shape[0]} vs {num_assets_passed}). Falling back to EW."
                )
                target_weights = equal_weights_policy(num_assets_passed)

            _heuristic_episode_cache[cache_key] = target_weights

        if target_weights.size == 0:  # Should not happen if EW fallback works
            log.warning(
                f"Heuristic {strategy_name} resulted in empty target_weights. Choosing random action."
            )
            return random.choice(np.arange(num_assets_passed or 1))

        action_index = int(np.argmax(target_weights))

        # Clear cache if it's the start of a new episode for this user
        # This simple cache clears when a new user/episode starts for a heuristic type.
        # A more robust cache would be passed in and managed by evaluate_policy.
        # For now, this module-level cache is per-user, per-step of the *same strategy*.
        # If evaluate_policy resets env, this cache key will be different for new episode.

        return action_index

    # This is a hacky way to clear cache. A better way is to have evaluate_policy manage it.
    # For now, let's clear it when get_heuristic_policy_decision_func is called,
    # assuming this is done once per heuristic evaluation run.
    _heuristic_episode_cache.clear()
    log.debug(f"Heuristic episode cache cleared for new call to {strategy_name}")

    return _heuristic_decision_maker
