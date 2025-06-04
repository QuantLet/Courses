# src/heuristics/utils.py
import numpy as np
import pandas as pd
import logging
from typing import Tuple, TYPE_CHECKING, List

# For type hinting to avoid circular imports
if TYPE_CHECKING:
    from environment.financial_env import FinancialRecEnv  # Assuming flat src structure

log = logging.getLogger(__name__)


def normalise_weights(weights: np.ndarray) -> np.ndarray:
    """
    Normalizes a weight vector to sum to 1, ensuring non-negativity.
    If all weights are zero or negative, returns an equal weight distribution.
    """
    w_non_negative = np.maximum(weights, 0)  # Ensure non-negative weights
    s = np.sum(w_non_negative)
    if s > 1e-9:  # Use a small threshold for numerical stability
        return w_non_negative / s
    else:
        log.debug(
            "Sum of non-negative weights is near zero in normalise_weights. Returning equal weights."
        )
        return np.full_like(weights, 1.0 / len(weights), dtype=float)


def calculate_mu_sigma_hat(
    prices_df: pd.DataFrame,
    asset_isins_ordered: List[str],  # To ensure output mu/sigma match asset order
    current_date: pd.Timestamp,
    lookback_days: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates estimated mean daily log returns (mu_hat) and covariance matrix
    of daily log returns (Sigma_hat) from historical daily prices.

    Args:
        prices_df: DataFrame with 'timestamp', 'ISIN', 'closePrice'.
                   MUST have 'timestamp' as DateTimeIndex if filtering by date range.
        asset_isins_ordered: List of ISINs in the desired order for mu_hat and Sigma_hat.
        current_date: The current date in the simulation (end of lookback period).
        lookback_days: Number of historical *price points* to use. (N days gives N-1 returns).

    Returns:
        Tuple (mu_hat, Sigma_hat). Returns (zeros, identity_matrix * 1e-6) if insufficient data.
    """
    num_assets = len(asset_isins_ordered)
    default_mu = np.zeros(num_assets)
    default_sigma = np.eye(num_assets) * 1e-6  # Small value for invertibility

    if prices_df.empty or num_assets == 0:
        log.warning(
            "calculate_mu_sigma_hat: Price data is empty or no assets specified. Returning default forecast."
        )
        return default_mu, default_sigma

    # Ensure prices_df has a DateTimeIndex for efficient slicing if not already
    # This depends on how prices_df is prepared before being passed.
    # For now, assume it might need conversion or is already indexed.
    # If it's not indexed, filtering is slower.
    # Original train.py assumed prices_df index was datetime when filtering.
    # Let's make it robust by pivoting and reindexing if necessary.

    # Pivot prices to have ISINs as columns and timestamp as index
    try:
        prices_pivot = prices_df.pivot(
            index="timestamp", columns="ISIN", values="closePrice"
        )
        # Reindex to ensure all assets in asset_isins_ordered are present, fill missing with NaN
        prices_pivot = prices_pivot.reindex(columns=asset_isins_ordered)
    except Exception as e:
        log.error(
            f"Error pivoting prices_df for mu/sigma calculation: {e}", exc_info=True
        )
        return default_mu, default_sigma

    end_date_dt = pd.to_datetime(current_date)
    # We need 'lookback_days' price points. So data from (end_date - (lookback_days-1)*days_offset) to end_date
    # Using business days might be more appropriate if prices are only on business days.
    # For simplicity with daily data as in mock:
    start_date_dt = end_date_dt - pd.DateOffset(
        days=lookback_days - 1
    )  # Correct for N price points

    # Filter price history for the lookback window
    # Ensure the index of prices_pivot is sorted for slicing
    if not prices_pivot.index.is_monotonic_increasing:
        prices_pivot = prices_pivot.sort_index()

    price_history_slice_df = prices_pivot[
        (prices_pivot.index >= start_date_dt) & (prices_pivot.index <= end_date_dt)
    ]
    # Fill NaNs that might arise from reindexing or missing data BEFORE taking .values
    # Forward fill first, then backfill for assets with no data at start/end of window.
    price_history_slice_df = price_history_slice_df.ffill().bfill()

    # Check if any asset has all NaNs after filling (means no price data at all for it in window)
    if price_history_slice_df.isna().all().any():
        log.warning(
            f"One or more assets have no price data in the lookback window ({start_date_dt.date()} to {end_date_dt.date()}) even after fill. Returning default forecast."
        )
        return default_mu, default_sigma

    if len(price_history_slice_df) < 2:  # Need at least 2 price points for 1 return
        log.warning(
            f"Not enough price history ({len(price_history_slice_df)} days) "
            f"for lookback ({lookback_days}) ending on {end_date_dt.strftime('%Y-%m-%d')}. "
            f"Need at least 2 days for 1 return. Returning default forecast."
        )
        return default_mu, default_sigma

    # Calculate daily log returns
    # .values converts to NumPy array. Columns are in order of asset_isins_ordered due to reindex.
    log_prices = np.log(price_history_slice_df.values)
    if not np.isfinite(log_prices).all():  # Check after log, before diff
        log.warning(
            "Non-finite log prices (e.g., from price=0 or negative). Returning default forecast."
        )
        return default_mu, default_sigma

    log_returns = np.diff(log_prices, axis=0)  # (N-1) x num_assets matrix

    if not np.isfinite(log_returns).all():
        log.warning("Non-finite log returns calculated. Returning default forecast.")
        return default_mu, default_sigma

    if log_returns.shape[0] == 0:  # Not enough data for any returns
        log.warning(
            "Zero log returns calculated (e.g. only 1 price point). Returning default forecast."
        )
        return default_mu, default_sigma

    mu_hat = np.mean(log_returns, axis=0)  # 1 x num_assets

    # Covariance calculation
    if log_returns.shape[0] < 2:  # Need at least 2 returns for meaningful covariance
        log.warning(
            f"Only {log_returns.shape[0]} returns available. Cannot calculate meaningful covariance. Using diagonal covariance."
        )
        # Use variance if available, else small identity
        asset_variances = (
            np.var(log_returns, axis=0)
            if log_returns.shape[0] > 0
            else np.zeros(num_assets)
        )
        sigma_hat = (
            np.diag(asset_variances) + np.eye(num_assets) * 1e-9
        )  # Small regularization
    else:
        # rowvar=False because each column is a variable (asset), rows are observations (days)
        # However, np.cov default is rowvar=True (each row is a variable).
        # So, if log_returns is (num_days x num_assets), we need to transpose it.
        sigma_hat = np.cov(log_returns.T)  # Transpose to (num_assets x num_days)
        # Add small diagonal perturbation for numerical stability
        sigma_hat = sigma_hat + np.eye(sigma_hat.shape[0]) * 1e-9

    # Ensure shapes are correct
    if mu_hat.shape != (num_assets,):
        mu_hat = np.zeros(num_assets)  # Fallback
        log.error("mu_hat shape mismatch, falling back to zeros.")
    if sigma_hat.shape != (num_assets, num_assets):
        sigma_hat = np.eye(num_assets) * 1e-6  # Fallback
        log.error("sigma_hat shape mismatch, falling back to identity.")

    return mu_hat, sigma_hat


def get_user_risk_budget_from_env(env: "FinancialRecEnv", user_id: str) -> float:
    """
    Retrieves the risk budget (sigma_budget) for a given user ID
    from the environment's pre-calculated user configurations.
    """
    if env.user_configs_for_heuristics.empty:
        log.warning(
            "Environment's user_configs_for_heuristics is empty. Returning default risk budget 0.1."
        )
        return 0.1

    user_row = env.user_configs_for_heuristics[
        env.user_configs_for_heuristics["User"] == user_id
    ]
    if user_row.empty:
        log.warning(
            f"Risk budget not found for user_id: {user_id} in env.user_configs. "
            f"Returning median budget or default 0.1."
        )
        # Fallback to median of available budgets or a hardcoded default
        if not env.user_configs_for_heuristics["σ_budget"].empty:
            return env.user_configs_for_heuristics["σ_budget"].median()
        return 0.1
    return float(user_row.iloc[0]["σ_budget"])


def get_current_date_from_env(env: "FinancialRecEnv") -> pd.Timestamp:
    """
    Retrieves the current simulation date from the environment.
    This is the date up to which heuristics should consider historical data.
    """
    # env.current_time is the start of the current month/step.
    # Heuristics often make decisions for the upcoming period based on data *up to* this point.
    return env.current_time
