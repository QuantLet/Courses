# src/financial_rl_project/data_management/mock_data.py
import numpy as np
import pandas as pd
from datetime import timedelta  # pd.Timestamp.now() handles datetime internally
import matplotlib.pyplot as plt  # For the __main__ block plotting
import seaborn as sns  # For the __main__ block plotting
import logging
from typing import Tuple

log = logging.getLogger(__name__)

# --- Default Configuration for Mock Data (can be overridden by function args from Hydra cfg) ---
N_ASSETS_MOCK_DEFAULT = 4
N_USERS_MOCK_DEFAULT = 3
PRICE_START_MOCK_DEFAULT = 100.0
TOTAL_DAYS_MOCK_DEFAULT = 4 * 365
ANNUAL_MU_RANGE_MOCK_DEFAULT = (0.03, 0.10)
ANNUAL_VOL_RANGE_MOCK_DEFAULT = (0.10, 0.30)
DAILY_TRADING_DAYS_DEFAULT = 252
DATASET_TYPE_DEFAULT = "training"


def _random_pos_sem_def(n_dim: int, rng_instance: np.random.Generator) -> np.ndarray:
    """Generates a random positive semi-definite matrix using a given RNG instance."""
    A = rng_instance.standard_normal((n_dim, n_dim))
    # Ensure it's at least positive semi-definite; for positive definite, add small identity
    # return A @ A.T + np.eye(n_dim) * 1e-6
    return A @ A.T


def create_mock_data(
    seed: int,
    n_assets: int = N_ASSETS_MOCK_DEFAULT,
    n_users: int = N_USERS_MOCK_DEFAULT,
    price_start: float = PRICE_START_MOCK_DEFAULT,
    total_days: int = TOTAL_DAYS_MOCK_DEFAULT,
    annual_mu_range: Tuple[float, float] = ANNUAL_MU_RANGE_MOCK_DEFAULT,
    annual_vol_range: Tuple[float, float] = ANNUAL_VOL_RANGE_MOCK_DEFAULT,
    daily_trading_days: int = DAILY_TRADING_DAYS_DEFAULT,
    dataset_type: str = DATASET_TYPE_DEFAULT,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generates mock DataFrames for customers, assets, prices, and transactions.

    Args:
        seed: Random seed for reproducibility.
        n_assets: Number of mock assets to generate.
        n_users: Number of mock users to generate.
        price_start: Initial price for asset time series.
        total_days: Total number of days for price history.
        annual_mu_range: Tuple (min, max) for annual drift of asset prices.
        annual_vol_range: Tuple (min, max) for annual volatility of asset prices.
        daily_trading_days: Number of trading days in a year for parameter conversion.
        dataset_type: Identifier ('training', 'validation') to slightly vary data.

    Returns:
        Tuple of DataFrames: (customers_df, assets_df, prices_df, transactions_df)
    """
    log.info(
        f"Generating mock data (seed={seed}, type='{dataset_type}', n_assets={n_assets}, n_users={n_users})."
    )
    rng = np.random.default_rng(seed)

    # 1. Customers DataFrame
    # Using the structure from your original mock_data.py (test_env.py)
    actual_n_users_for_type = n_users
    customer_id_prefix = "C"
    if (
        dataset_type.lower() == "validation"
    ):  # Slightly different data for validation set
        actual_n_users_for_type = n_users + 1  # Example: one more user
        customer_id_prefix = "VC"
        default_risk_levels = ["Aggressive", "Balanced", "Conservative", "Income"]
        default_inv_caps = ["CAP_GT300K", "CAP_80K_300K", "CAP_30K_80K", "CAP_LT30K"]
    else:  # Training data
        customer_id_prefix = "TC"
        default_risk_levels = [
            "Balanced",
            "Aggressive",
            "Conservative",
            "Income",
        ]  # Provide enough variety
        default_inv_caps = ["CAP_80K_300K", "CAP_GT300K", "CAP_30K_80K", "CAP_LT30K"]

    customers_data = {
        "customerID": [
            f"{customer_id_prefix}{i + 1}" for i in range(actual_n_users_for_type)
        ],
        "riskLevel": [
            default_risk_levels[i % len(default_risk_levels)]
            for i in range(actual_n_users_for_type)
        ],
        "investmentCapacity": [
            default_inv_caps[i % len(default_inv_caps)]
            for i in range(actual_n_users_for_type)
        ],
    }
    customers_df = pd.DataFrame(customers_data)

    # 2. Assets DataFrame
    asset_isins = [f"ISIN_MOCK_{chr(65 + i)}" for i in range(n_assets)]
    assets_data = {
        "ISIN": asset_isins,
        # VolatilityQuartile: 1,2,3,4 (raw quartile) as in your original script.
        # The environment's _calculate_risk_match_score should handle mapping this (e.g., quartile - 1).
        "VolatilityQuartile": rng.integers(1, 5, size=n_assets),
    }
    assets_df = pd.DataFrame(assets_data)

    # 3. Transactions DataFrame (Historical)
    fixed_today_ref = pd.Timestamp(
        "2024-01-01"
    )  # Use a fixed "today" for reproducibility of dates
    num_txns = actual_n_users_for_type * rng.integers(
        3, 7
    )  # Each user has 3-6 transactions

    # Ensure transaction timestamps are within the price history range
    max_txn_days_ago = (
        total_days - 30
    )  # Ensure txns are not on the last day of price history
    min_txn_days_ago = 1

    txn_timestamps = fixed_today_ref - pd.to_timedelta(
        rng.uniform(min_txn_days_ago, max_txn_days_ago, size=num_txns), unit="D"
    )
    transactions_data = {
        "customerID": rng.choice(customers_df["customerID"], size=num_txns).tolist(),
        "ISIN": rng.choice(assets_df["ISIN"], size=num_txns).tolist(),
        "transactionType": rng.choice(["Buy", "Sell"], size=num_txns).tolist(),
        "totalValue": rng.uniform(1000, 20000, size=num_txns).round(2),
        "timestamp": txn_timestamps,
    }
    transactions_df = pd.DataFrame(transactions_data)
    transactions_df["timestamp"] = pd.to_datetime(
        transactions_df["timestamp"]
    ).dt.normalize()  # Normalize to midnight
    transactions_df = transactions_df.sort_values("timestamp").reset_index(drop=True)

    # 4. Close Prices DataFrame
    log.debug("Simulating correlated daily closing prices for mock data...")
    price_start_date = fixed_today_ref - pd.DateOffset(days=total_days - 1)
    price_date_range = pd.date_range(
        start=price_start_date, end=fixed_today_ref, freq="D"
    )
    n_price_days = len(price_date_range)

    annual_mus_arr = rng.uniform(annual_mu_range[0], annual_mu_range[1], n_assets)
    annual_sigmas_arr = rng.uniform(annual_vol_range[0], annual_vol_range[1], n_assets)
    daily_mus_arr = annual_mus_arr / daily_trading_days
    daily_sigmas_arr = annual_sigmas_arr / np.sqrt(daily_trading_days)

    # Generate daily covariance matrix
    random_cov_matrix = _random_pos_sem_def(n_assets, rng)
    current_std_from_cov = np.sqrt(np.diag(random_cov_matrix))
    current_std_from_cov[current_std_from_cov < 1e-8] = (
        1e-8  # Avoid division by zero if std is too small
    )

    scaling_diag = daily_sigmas_arr / current_std_from_cov
    # sigma_daily = D * C * D where C is random_cov_matrix (as correlation proxy) and D is scaling_diag
    # This is not strictly correct if random_cov_matrix isn't a correlation matrix.
    # A more robust way if starting with a generic PSD matrix:
    # 1. Decompose random_cov_matrix = L L^T
    # 2. Create correlation matrix C from random_cov_matrix: C_ij = Cov_ij / (std_i * std_j)
    # 3. Target Covariance Sigma_target = diag(daily_sigmas_arr) @ C @ diag(daily_sigmas_arr)
    # For simplicity, let's use the approach from your original script if it worked:
    # It was scaling_matrix @ random_cov @ scaling_matrix
    # Assuming random_cov_matrix needs to be treated as a base for scaling
    sigma_daily_cov_matrix = (
        np.diag(scaling_diag) @ random_cov_matrix @ np.diag(scaling_diag)
    )

    # Ensure positive definiteness after scaling (numerical precision can be an issue)
    # Add a small diagonal epsilon if not PD.
    try:
        np.linalg.cholesky(sigma_daily_cov_matrix)
    except np.linalg.LinAlgError:
        log.warning(
            "Daily covariance matrix not positive definite after scaling, adding small identity perturbation."
        )
        sigma_daily_cov_matrix += np.eye(n_assets) * 1e-9  # Small epsilon

    # Simulate daily log returns
    shocks = rng.multivariate_normal(
        np.zeros(n_assets), sigma_daily_cov_matrix, size=n_price_days
    )
    daily_log_returns = daily_mus_arr + shocks  # Add daily mean drift

    # Simulate prices
    price_history = np.zeros((n_price_days, n_assets))
    initial_prices = np.full(n_assets, price_start)
    price_history[0] = initial_prices

    for i in range(1, n_price_days):
        price_history[i] = price_history[i - 1] * np.exp(daily_log_returns[i])
        price_history[i] = np.maximum(price_history[i], 0.01)  # Floor prices at 0.01

    price_data_list = []
    for i_day, date_val in enumerate(price_date_range):
        for j_asset, isin_val in enumerate(asset_isins):
            price_data_list.append(
                {
                    "ISIN": isin_val,
                    "timestamp": date_val,  # Already normalized
                    "closePrice": price_history[i_day, j_asset],
                }
            )
    prices_df = pd.DataFrame(price_data_list)
    # prices_df['timestamp'] is already pd.to_datetime from date_range

    log.info("Mock data generation complete.")
    return customers_df, assets_df, prices_df, transactions_df


# --- Main Execution Block for Plotting (if run directly) ---
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
    )
    log.info("--- Running mock_data.py script directly for testing/plotting ---")

    test_customers_df, test_assets_df, test_prices_df, test_transactions_df = (
        create_mock_data(seed=42)
    )

    log.info("\nMock Data Shapes:")
    log.info(f"  Customers: {test_customers_df.shape}")
    log.info(f"  Assets: {test_assets_df.shape}")
    log.info(f"  Transactions: {test_transactions_df.shape}")
    log.info(f"  Prices: {test_prices_df.shape}")

    log.info("\nGenerating time series plot of mock prices...")
    price_pivot_for_plot = test_prices_df.pivot(
        index="timestamp", columns="ISIN", values="closePrice"
    )
    plt.figure(figsize=(14, 7))
    sns.lineplot(data=price_pivot_for_plot)
    plt.title("Simulated Mock Asset Prices")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend(title="Asset ISIN", loc="upper left")
    plt.tight_layout()
    # Save plot if needed: plt.savefig("mock_prices_simulation.png")
    plt.show()

    log.info("--- mock_data.py script finished ---")
