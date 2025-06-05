# src/financial_rl_project/environment/financial_env.py
import numpy as np
import pandas as pd
import logging
from typing import Tuple, Dict, Any, List, Optional

# Configure logging for this module
log = logging.getLogger(__name__)

# Environment-specific constants
RISK_MAP = {"Income": 0, "Conservative": 1, "Balanced": 2, "Aggressive": 3}
DEFAULT_RISK = RISK_MAP["Balanced"]
CAPACITY_MAP = {"CAP_LT30K": 0, "CAP_30K_80K": 1, "CAP_80K_300K": 2, "CAP_GT300K": 3}
DEFAULT_CAPACITY = CAPACITY_MAP["CAP_80K_300K"]


class FinancialRecEnv:
    """
    Financial Recommendation Environment for Reinforcement Learning.

    Simulates a financial advisor recommending assets to customers over time.
    The state includes customer risk/capacity and current portfolio allocation.
    The reward is based on price performance, risk match, and portfolio diversification.
    """

    def __init__(
        self,
        customers: pd.DataFrame,
        assets: pd.DataFrame,
        prices: pd.DataFrame,
        transactions: pd.DataFrame,
        initial_offset_months: int,
        episode_max_steps: int,
        accept_probability: float,
        reward_scale: float,
        reward_price_weight: float,
        reward_risk_weight: float,
        reward_hhi_weight: float,
    ):
        """
        Initializes the financial recommendation environment.

        Args:
            customers: DataFrame with customer information (customerID, riskLevel, investmentCapacity).
            assets: DataFrame with asset information (ISIN, VolatilityQuartile).
            prices: DataFrame with asset close prices (ISIN, timestamp, closePrice).
            transactions: DataFrame with historical transactions (customerID, ISIN, transactionType, totalValue, timestamp).
            initial_offset_months: Number of months before the latest data point to start the simulation.
            episode_max_steps: Maximum number of steps (months) per episode.
            accept_probability: Probability that a customer accepts a recommendation.
            reward_scale: Scaling factor for the total reward.
            reward_price_weight: Weight for the price performance component of the reward.
            reward_risk_weight: Weight for the risk match component of the reward.
            reward_hhi_weight: Weight for the HHI (diversification) component of the reward.
        """
        log.info("Initializing FinancialRecEnv...")

        if customers.empty or assets.empty or prices.empty or transactions.empty:
            log.error(
                "Input DataFrames (customers, assets, prices, transactions) cannot be empty."
            )
            raise ValueError("Input DataFrames cannot be empty.")

        self.customers_df = customers.copy()
        self.assets_df = assets.copy()
        self.transactions_df = transactions.copy()
        self.prices_df = prices.copy()

        # Validate and convert data types
        self._validate_and_prepare_data()

        self.initial_offset_months = initial_offset_months
        self.episode_max_steps = episode_max_steps
        self.accept_probability = accept_probability
        self.reward_scale = reward_scale
        self.reward_price_weight = reward_price_weight
        self.reward_risk_weight = reward_risk_weight
        self.reward_hhi_weight = reward_hhi_weight

        # --- Action Space ---
        self.asset_isins_list: List[str] = sorted(
            self.assets_df["ISIN"].unique().tolist()
        )
        if not self.asset_isins_list:
            log.error("No unique ISINs found in assets_df to define an action space.")
            raise ValueError("No unique ISINs found in assets to define action space.")
        self.num_actions: int = len(self.asset_isins_list)
        self.action_to_index: Dict[str, int] = {
            isin: i for i, isin in enumerate(self.asset_isins_list)
        }
        self.index_to_action: Dict[int, str] = {
            i: isin for i, isin in enumerate(self.asset_isins_list)
        }

        # --- Time Management ---
        self.min_data_timestamp = min(
            self.transactions_df["timestamp"].min(), self.prices_df["timestamp"].min()
        )
        self.max_data_timestamp = max(
            self.transactions_df["timestamp"].max(), self.prices_df["timestamp"].max()
        )

        if self.min_data_timestamp >= self.max_data_timestamp:
            log.error(
                f"Min data time {self.min_data_timestamp} is not before max data time {self.max_data_timestamp}."
            )
            raise ValueError("Min data time is not before max data time.")

        # Simulation start time
        self.simulation_start_time = self.max_data_timestamp - pd.DateOffset(
            months=self.initial_offset_months
        )
        if self.simulation_start_time < self.min_data_timestamp:
            log.warning(
                f"Calculated simulation_start_time ({self.simulation_start_time}) is before the earliest data "
                f"({self.min_data_timestamp}). Starting simulation from {self.min_data_timestamp}."
            )
            self.simulation_start_time = self.min_data_timestamp

        # Monthly time index for simulation steps and heuristic lookups
        self.time_index_monthly = pd.date_range(
            start=self.min_data_timestamp.normalize(),  # Use actual min data time
            end=(
                self.max_data_timestamp + pd.DateOffset(months=1)
            ).normalize(),  # Go one month past for stepping
            freq="MS",  # Month Start frequency
        )

        self.current_time: pd.Timestamp = self.simulation_start_time
        self.current_month_idx: int = 0  # Index into self.time_index_monthly

        # --- State Dimension ---
        # [risk_level (numeric), investment_capacity (numeric), asset_0_pct, ..., asset_N-1_pct]
        self.state_dimension: int = (
            2 + self.num_actions
        )  # 2 for customer profile features

        # --- Episode State ---
        self.current_customer_id: Optional[str] = None
        self.current_episode_step: int = 0

        # For heuristic compatibility (get_user_risk_budget)
        self.user_configs_for_heuristics = self.customers_df[
            ["customerID", "riskLevel"]
        ].copy()
        risk_to_budget_map = {
            "Income": 0.05,
            "Conservative": 0.10,
            "Balanced": 0.15,
            "Aggressive": 0.20,
        }
        self.user_configs_for_heuristics["σ_budget"] = (
            self.user_configs_for_heuristics["riskLevel"]
            .map(risk_to_budget_map)
            .fillna(0.15)
        )
        # Rename customerID to User to match heuristic function's expectation
        self.user_configs_for_heuristics.rename(
            columns={"customerID": "User"}, inplace=True
        )

        log.info("FinancialRecEnv initialized successfully.")
        log.info(
            f"  State Dimension: {self.state_dimension}, Action Space Size: {self.num_actions}"
        )
        log.info(
            f"  Data Timestamp Range: {self.min_data_timestamp.strftime('%Y-%m-%d')} to {self.max_data_timestamp.strftime('%Y-%m-%d')}"
        )
        log.info(
            f"  Simulation Start Timestamp: {self.simulation_start_time.strftime('%Y-%m-%d')}"
        )

    def _validate_and_prepare_data(self):
        """Validates required columns and converts data types."""
        log.debug("Validating and preparing input data types...")
        try:
            # Customers
            req_cols_cust = ["customerID", "riskLevel", "investmentCapacity"]
            if not all(col in self.customers_df.columns for col in req_cols_cust):
                raise ValueError(
                    f"Customers DataFrame missing one or more required columns: {req_cols_cust}"
                )
            self.customers_df["customerID"] = self.customers_df["customerID"].astype(
                str
            )

            # Assets
            req_cols_assets = ["ISIN", "VolatilityQuartile"]
            if not all(col in self.assets_df.columns for col in req_cols_assets):
                raise ValueError(
                    f"Assets DataFrame missing one or more required columns: {req_cols_assets}"
                )
            self.assets_df["ISIN"] = self.assets_df["ISIN"].astype(str)
            self.assets_df["VolatilityQuartile"] = pd.to_numeric(
                self.assets_df["VolatilityQuartile"], errors="coerce"
            )
            # Handle assets with NaN VolatilityQuartile if necessary (e.g., assign a default or log warning)
            if self.assets_df["VolatilityQuartile"].isna().any():
                log.warning(
                    f"Found {self.assets_df['VolatilityQuartile'].isna().sum()} assets with NaN VolatilityQuartile."
                )
                # self.assets_df['VolatilityQuartile'].fillna(DEFAULT_RISK_QUARTILE_EQUIVALENT, inplace=True) # Example fill

            # Prices
            req_cols_prices = ["ISIN", "timestamp", "closePrice"]
            if not all(col in self.prices_df.columns for col in req_cols_prices):
                raise ValueError(
                    f"Prices DataFrame missing one or more required columns: {req_cols_prices}"
                )
            self.prices_df["ISIN"] = self.prices_df["ISIN"].astype(str)
            self.prices_df["timestamp"] = pd.to_datetime(
                self.prices_df["timestamp"], errors="coerce"
            )
            self.prices_df["closePrice"] = pd.to_numeric(
                self.prices_df["closePrice"], errors="coerce"
            )
            self.prices_df.dropna(
                subset=["timestamp", "closePrice"], inplace=True
            )  # Critical columns

            # Transactions
            req_cols_txns = [
                "customerID",
                "ISIN",
                "transactionType",
                "totalValue",
                "timestamp",
            ]
            if not all(col in self.transactions_df.columns for col in req_cols_txns):
                raise ValueError(
                    f"Transactions DataFrame missing one or more required columns: {req_cols_txns}"
                )
            self.transactions_df["customerID"] = self.transactions_df[
                "customerID"
            ].astype(str)
            self.transactions_df["ISIN"] = self.transactions_df["ISIN"].astype(str)
            self.transactions_df["timestamp"] = pd.to_datetime(
                self.transactions_df["timestamp"], errors="coerce"
            )
            self.transactions_df["totalValue"] = pd.to_numeric(
                self.transactions_df["totalValue"], errors="coerce"
            )
            self.transactions_df.dropna(
                subset=["timestamp", "totalValue"], inplace=True
            )  # Critical columns

            # Sort prices by timestamp for efficient lookup if not already sorted
            self.prices_df.sort_values(by=["ISIN", "timestamp"], inplace=True)

        except Exception as e:
            log.error(
                f"Error during data validation and preparation: {e}", exc_info=True
            )
            raise

    def get_numerical_state(self, customer_id: str) -> np.ndarray:
        """
        Retrieves the numerical state vector for a given customer.
        State: [risk_level, investment_capacity, pct_asset_0, ..., pct_asset_N-1].
        """
        customer_profile_row = self.customers_df[
            self.customers_df["customerID"] == customer_id
        ]
        if customer_profile_row.empty:
            log.warning(
                f"Customer ID {customer_id} not found in customers_df. Returning zero state vector."
            )
            return np.zeros(self.state_dimension, dtype=np.float32)

        customer_profile = customer_profile_row.iloc[0]
        risk_level = RISK_MAP.get(customer_profile.get("riskLevel"), DEFAULT_RISK)
        invest_capacity = CAPACITY_MAP.get(
            customer_profile.get("investmentCapacity"), DEFAULT_CAPACITY
        )

        portfolio_values = self.get_customer_portfolio(customer_id, self.current_time)
        total_portfolio_value = sum(portfolio_values.values())

        portfolio_pct_vector = np.zeros(self.num_actions, dtype=np.float32)
        if total_portfolio_value > 1e-6:  # Avoid division by zero
            for isin, value in portfolio_values.items():
                if isin in self.action_to_index:
                    idx = self.action_to_index[isin]
                    portfolio_pct_vector[idx] = value / total_portfolio_value

        state_vector = np.concatenate(
            ([float(risk_level), float(invest_capacity)], portfolio_pct_vector)
        ).astype(np.float32)

        if len(state_vector) != self.state_dimension:
            log.error(
                f"State vector length mismatch! Expected {self.state_dimension}, got {len(state_vector)} for customer {customer_id}"
            )
            # This should ideally not happen, indicates an issue with num_actions or concatenation
            raise RuntimeError("Internal state vector dimension error.")
        return state_vector

    def reset(
        self, customer_id_to_set: Optional[str] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Resets the environment for a new episode.

        Args:
            customer_id_to_set: Specific customer ID to start the episode with.
                                If None, a random customer is chosen.
        Returns:
            Tuple of (initial_state_vector, info_dictionary).
        """
        if customer_id_to_set:
            if customer_id_to_set not in self.customers_df["customerID"].unique():
                log.warning(
                    f"Provided customer_id '{customer_id_to_set}' not found. Selecting a random customer."
                )
                self.current_customer_id = np.random.choice(
                    self.customers_df["customerID"].unique()
                )
            else:
                self.current_customer_id = customer_id_to_set
        else:
            self.current_customer_id = np.random.choice(
                self.customers_df["customerID"].unique()
            )

        self.current_time = self.simulation_start_time
        self.current_episode_step = 0

        # Set current_month_idx based on simulation_start_time relative to time_index_monthly
        # searchsorted returns the insertion point; -1 gives the index of the month start <= current_time
        self.current_month_idx = (
            self.time_index_monthly.searchsorted(self.current_time, side="right") - 1
        )
        if (
            self.current_month_idx < 0
        ):  # Should not happen if simulation_start_time >= min_data_timestamp
            self.current_month_idx = 0
        if (
            self.time_index_monthly[self.current_month_idx] > self.current_time
        ):  # If start_time is before first month_start
            if self.current_month_idx > 0:
                self.current_month_idx -= 1

        log.debug(
            f"Resetting environment for customer: {self.current_customer_id}, "
            f"Start time: {self.current_time.strftime('%Y-%m-%d')}, "
            f"Month_idx: {self.current_month_idx}"
        )

        initial_state = self.get_numerical_state(self.current_customer_id)
        info = {
            "customer_id": self.current_customer_id,
            "current_time": self.current_time.strftime("%Y-%m-%d"),
            "current_month_idx": self.current_month_idx,
            "episode_step": self.current_episode_step,
        }
        return initial_state, info

    def _advance_time(self) -> bool:
        """
        Advances the simulation time by one month.
        Returns:
            bool: True if the episode should terminate due to time limit, False otherwise.
        """
        self.current_month_idx += 1
        if self.current_month_idx >= len(self.time_index_monthly):
            log.debug(
                f"Time limit reached: current_month_idx ({self.current_month_idx}) "
                f"exceeds time_index_monthly length ({len(self.time_index_monthly)})."
            )
            return True  # Terminate: Ran out of predefined months

        self.current_time = self.time_index_monthly[self.current_month_idx]

        # Additional check: if current_time significantly past max_data_timestamp
        if self.current_time > (self.max_data_timestamp + pd.DateOffset(days=10)):
            log.debug(
                f"Time limit reached: current_time ({self.current_time}) "
                f"is past max_data_timestamp ({self.max_data_timestamp})."
            )
            return True  # Terminate
        return False

    def step(
        self, action_index: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Executes one time step in the environment based on the selected action.

        Args:
            action_index: Index of the asset chosen by the agent.

        Returns:
            Tuple of (next_state, reward, terminated, truncated, info_dictionary).
        """
        if self.current_customer_id is None:
            log.error("Environment must be reset before calling step().")
            raise RuntimeError("Environment not reset. Call reset() first.")

        terminated = False
        truncated = False
        info = {"customer_id": self.current_customer_id}
        reward = 0.0

        if not (0 <= action_index < self.num_actions):
            log.warning(
                f"Invalid action_index {action_index} received. Max actions: {self.num_actions - 1}. "
                f"Taking no action, 0 reward."
            )
            action_isin = "INVALID_ACTION"  # For logging
            info.update(
                {
                    "action_isin": action_isin,
                    "recommendation_accepted": False,
                    "transaction_added": False,
                }
            )
        else:
            action_isin = self.index_to_action[action_index]
            info["action_isin"] = action_isin

            current_reward = self.calculate_reward(
                self.current_customer_id, action_isin, self.current_time
            )

            recommendation_accepted = False
            transaction_added = False
            if np.random.rand() < self.accept_probability:
                recommendation_accepted = True
                transaction_value = self._get_simulated_transaction_value(
                    self.current_customer_id
                )

                # Add new transaction using current_time (which is start of the month)
                new_transaction_data = {
                    "customerID": [self.current_customer_id],
                    "ISIN": [action_isin],
                    "transactionType": ["Buy"],
                    "totalValue": [transaction_value],
                    "timestamp": [
                        self.current_time
                    ],  # Transaction occurs at the current simulation time
                }
                new_transaction_df = pd.DataFrame(new_transaction_data)
                self.transactions_df = pd.concat(
                    [self.transactions_df, new_transaction_df], ignore_index=True
                )
                transaction_added = True
                current_reward += 0.1  # Small bonus for accepted transaction (optional)
                log.debug(
                    f"Customer {self.current_customer_id} accepted recommendation for {action_isin}, value {transaction_value:.2f}"
                )
            else:
                current_reward -= (
                    0.05  # Small penalty for rejected recommendation (optional)
                )
                log.debug(
                    f"Customer {self.current_customer_id} rejected recommendation for {action_isin}"
                )

            reward = current_reward
            info.update(
                {
                    "recommendation_accepted": recommendation_accepted,
                    "transaction_added": transaction_added,
                }
            )

        # Advance time and episode step
        time_terminated = self._advance_time()
        self.current_episode_step += 1

        next_state = self.get_numerical_state(self.current_customer_id)

        terminated = time_terminated  # Episode ends if time limit is reached
        if self.current_episode_step >= self.episode_max_steps:
            truncated = True  # Episode ends if max steps reached
            log.debug(
                f"Episode truncated at step {self.current_episode_step} for customer {self.current_customer_id}."
            )
        current_portfolio_details = self.get_customer_portfolio(
            self.current_customer_id, self.current_time
        )
        current_portfolio_value = sum(current_portfolio_details.values())
        current_portfolio_hhi = self._calculate_hhi(current_portfolio_details)

        info.update(
            {
                "current_time": self.current_time.strftime("%Y-%m-%d"),
                "current_month_idx": self.current_month_idx,
                "episode_step": self.current_episode_step,
                "portfolio_value": current_portfolio_value,
                "portfolio_hhi": current_portfolio_hhi,
                "portfolio_composition": current_portfolio_details,  # {ISIN: value}
            }
        )
        # Ensure reward is a standard float
        reward = float(np.nan_to_num(reward))

        return next_state, reward, terminated, truncated, info

    def _get_simulated_transaction_value(self, customer_id: str) -> float:
        """Determines a plausible transaction value based on customer's investment capacity."""
        customer_profile = self.customers_df[
            self.customers_df["customerID"] == customer_id
        ]
        if customer_profile.empty:
            log.warning(
                f"Customer {customer_id} not found for transaction value simulation. Using default range."
            )
            capacity_str = None
        else:
            capacity_str = customer_profile.iloc[0].get("investmentCapacity")

        # Define value ranges based on capacity string
        if capacity_str == "CAP_LT30K":
            min_val, max_val = 50, 500
        elif capacity_str == "CAP_30K_80K":
            min_val, max_val = 200, 2000
        elif capacity_str == "CAP_80K_300K":
            min_val, max_val = 1000, 10000
        elif capacity_str == "CAP_GT300K":
            min_val, max_val = 5000, 50000
        else:  # Default or unknown capacity
            min_val, max_val = 100, 1000
        return float(np.random.uniform(min_val, max_val))

    def _get_value_range_for_capacity(self, customer_id: str) -> Tuple[float, float]:
        """Helper for reward calculation, returns min/max purchase value for capacity."""
        # This is largely duplicated by _get_simulated_transaction_value logic
        # Consolidating logic or ensuring consistency is good.
        customer_profile = self.customers_df[
            self.customers_df["customerID"] == customer_id
        ]
        capacity_str = (
            customer_profile.iloc[0].get("investmentCapacity")
            if not customer_profile.empty
            else None
        )

        if capacity_str == "CAP_LT30K":
            return 50, 500
        elif capacity_str == "CAP_30K_80K":
            return 200, 2000
        elif capacity_str == "CAP_80K_300K":
            return 1000, 10000
        elif capacity_str == "CAP_GT300K":
            return 5000, 50000
        else:
            return 100, 1000

    def _calculate_risk_match_score(
        self, customer_risk_level: int, asset_volatility_quartile_1_4: Optional[float]
    ) -> float:
        """
        Calculates risk match score.
        Assumes asset_volatility_quartile_1_4 is 1, 2, 3, or 4.
        Compares with customer_risk_level (0-3 from RISK_MAP).
        """
        if asset_volatility_quartile_1_4 is None or np.isnan(
            asset_volatility_quartile_1_4
        ):
            return 0.0  # Neutral score if asset risk is unknown

        # Convert asset's 1-4 quartile to 0-3 scale for comparison
        asset_risk_level_0_3 = float(asset_volatility_quartile_1_4) - 1.0

        if customer_risk_level == asset_risk_level_0_3:
            return 1.0  # Perfect match
        elif abs(customer_risk_level - asset_risk_level_0_3) == 1:
            return 0.0  # Close match (neutral or small positive)
        else:
            return -1.0  # Mismatch

    def calculate_reward(
        self, customer_id: str, recommended_isin: str, at_time: pd.Timestamp
    ) -> float:
        """
        Computes the reward for recommending a given ISIN to a customer at a specific time.
        """
        asset_info_row = self.assets_df.loc[self.assets_df["ISIN"] == recommended_isin]
        customer_info_row = self.customers_df.loc[
            self.customers_df["customerID"] == customer_id
        ]

        if asset_info_row.empty or customer_info_row.empty:
            log.warning(
                f"Asset {recommended_isin} or Customer {customer_id} not found for reward calculation. Returning 0 reward."
            )
            return 0.0

        asset_profile = asset_info_row.iloc[0]
        customer_profile = customer_info_row.iloc[0]
        customer_risk_level_mapped = RISK_MAP.get(
            customer_profile["riskLevel"], DEFAULT_RISK
        )

        # 1. Performance Component (Price change over the last month from at_time)
        price_reference_time = at_time - pd.DateOffset(months=1)
        performance_score = self.price_difference(
            recommended_isin, price_reference_time, at_time
        )

        # 2. Risk Match Component
        asset_vol_quartile_1_4 = asset_profile.get(
            "VolatilityQuartile"
        )  # Expected to be 1,2,3,4 or NaN
        risk_match_score = self._calculate_risk_match_score(
            customer_risk_level_mapped, asset_vol_quartile_1_4
        )

        # 3. Diversification Component (Change in HHI)
        current_portfolio = self.get_customer_portfolio(
            customer_id, at_time
        )  # Portfolio before this month's recommendation
        current_hhi = self._calculate_hhi(current_portfolio)

        # Estimate portfolio *after* hypothetical purchase of recommended_isin
        min_val, max_val = self._get_value_range_for_capacity(customer_id)
        estimated_purchase_value = (
            min_val + max_val
        ) / 2.0  # Use average for estimation

        hypothetical_next_portfolio = current_portfolio.copy()
        hypothetical_next_portfolio[recommended_isin] = (
            hypothetical_next_portfolio.get(recommended_isin, 0.0)
            + estimated_purchase_value
        )
        next_hhi = self._calculate_hhi(hypothetical_next_portfolio)

        hhi_change = next_hhi - current_hhi  # Negative if HHI decreased (good)
        # Scale diversification score: make improvements more impactful. Max HHI change is 1.
        # A change of -0.1 (e.g. 0.5 to 0.4) could become a score of +0.5 if scaled by * -5
        diversification_score = -hhi_change * 5.0

        # Combine reward components
        reward = (
            self.reward_price_weight * performance_score
            + self.reward_risk_weight * risk_match_score
            + self.reward_hhi_weight * diversification_score
        )
        scaled_reward = reward * self.reward_scale

        # Sanity checks for reward value
        if np.isnan(scaled_reward) or np.isinf(scaled_reward):
            log.warning(
                f"Reward calculation for C:{customer_id} A:{recommended_isin} resulted in NaN/Inf. "
                f"Scores: Perf={performance_score:.2f}, Risk={risk_match_score:.2f}, Div={diversification_score:.2f}. Setting reward to 0."
            )
            return 0.0

        # Optional: clip reward to a reasonable range if experiencing extreme values
        # scaled_reward = np.clip(scaled_reward, -abs(self.reward_scale * 3), abs(self.reward_scale * 3))

        return float(scaled_reward)

    def get_customer_portfolio(
        self, customer_id: str, at_time: pd.Timestamp
    ) -> Dict[str, float]:
        """
        Calculates the customer's portfolio (holdings value per ISIN) up to a given time.
        """
        # Ensure timestamps are compatible for comparison (e.g., both timezone-naive or same timezone)
        # This should ideally be handled during data loading by standardizing all timestamps (e.g., to UTC or naive).
        # Assuming self.transactions_df['timestamp'] and at_time are compatible here.

        relevant_transactions = self.transactions_df[
            (self.transactions_df["customerID"] == customer_id)
            & (
                self.transactions_df["timestamp"] <= at_time
            )  # Only transactions up to 'at_time'
        ].copy()

        if relevant_transactions.empty:
            return {}

        # Determine effect of transaction type on value
        relevant_transactions["transactionEffect"] = relevant_transactions[
            "transactionType"
        ].apply(
            lambda x: 1 if x.lower() == "buy" else (-1 if x.lower() == "sell" else 0)
        )
        relevant_transactions["adjustedValue"] = (
            relevant_transactions["totalValue"]
            * relevant_transactions["transactionEffect"]
        )

        portfolio_series = relevant_transactions.groupby("ISIN")["adjustedValue"].sum()

        # Filter out assets with zero or negative net holdings
        current_holdings = portfolio_series[
            portfolio_series > 1e-6
        ].to_dict()  # Use small epsilon
        return current_holdings

    def _calculate_hhi(self, portfolio_holdings: Dict[str, float]) -> float:
        """
        Calculates the Herfindahl-Hirschman Index (HHI) for a portfolio.
        Lower HHI indicates more diversification. Max HHI is 1.0 (single asset).
        """
        total_portfolio_value = sum(portfolio_holdings.values())
        if (
            total_portfolio_value < 1e-6
        ):  # Avoid division by zero for empty or zero-value portfolio
            return 1.0  # Max HHI (completely undiversified)

        hhi = sum(
            [
                (value / total_portfolio_value) ** 2
                for value in portfolio_holdings.values()
            ]
        )
        return hhi

    def price_difference(
        self, isin: str, past_time: pd.Timestamp, current_time_env: pd.Timestamp
    ) -> float:
        """
        Calculates the relative price difference: (price(current) - price(past)) / price(past).
        Uses the latest available price at or before the specified times.
        """
        asset_prices = self.prices_df[
            self.prices_df["ISIN"] == isin
        ]  # Already sorted by timestamp in __init__
        if asset_prices.empty:
            log.debug(
                f"No price data found for ISIN {isin} for price_difference calculation."
            )
            return 0.0

        # Find price at past_time (or latest before it)
        past_price_idx = asset_prices["timestamp"].searchsorted(past_time, side="right")
        if past_price_idx == 0:  # No data at or before past_time
            price_past = np.nan
        else:
            price_past = asset_prices.iloc[past_price_idx - 1]["closePrice"]

        # Find price at current_time_env (or latest before it)
        current_price_idx = asset_prices["timestamp"].searchsorted(
            current_time_env, side="right"
        )
        if current_price_idx == 0:  # No data at or before current_time_env
            price_current = np.nan
        else:
            price_current = asset_prices.iloc[current_price_idx - 1]["closePrice"]

        if pd.isna(price_past) or pd.isna(price_current) or price_past == 0:
            log.debug(
                f"Could not calculate price difference for ISIN {isin}: "
                f"PastPrice={price_past}, CurrentPrice={price_current} between {past_time} and {current_time_env}."
            )
            return 0.0

        relative_diff = (price_current - price_past) / price_past
        return float(
            np.nan_to_num(relative_diff)
        )  # Handle potential NaNs from operations if any


# Example for testing this module directly (optional)
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
    )
    log.info("Testing FinancialRecEnv module...")

    # Create minimal mock data for testing
    mock_customers = pd.DataFrame(
        {
            "customerID": ["C1", "C2"],
            "riskLevel": ["Balanced", "Aggressive"],
            "investmentCapacity": ["CAP_30K_80K", "CAP_GT300K"],
        }
    )
    mock_assets = pd.DataFrame(
        {
            "ISIN": ["A1", "A2", "A3"],
            "AssetName": [
                "Asset 1",
                "Asset 2",
                "Asset 3",
            ],  # AssetName not used by env core
            "VolatilityQuartile": [2, 3, 4],  # Example: Quartiles 1-4
        }
    )

    sim_start_date = pd.Timestamp("2022-01-01")
    price_dates = pd.date_range(
        sim_start_date - pd.DateOffset(years=3), periods=4 * 365, freq="D"
    )

    price_data_list = []
    for isin_mock in mock_assets["ISIN"]:
        prices_mock_ts = 100 + np.cumsum(np.random.randn(len(price_dates)) * 0.5 + 0.01)
        prices_mock_ts = np.maximum(prices_mock_ts, 1.0)
        for i_pr, date_pr in enumerate(price_dates):
            price_data_list.append(
                {
                    "timestamp": date_pr,
                    "ISIN": isin_mock,
                    "closePrice": prices_mock_ts[i_pr],
                }
            )
    mock_prices = pd.DataFrame(price_data_list)

    mock_transactions = pd.DataFrame(
        {
            "customerID": ["C1", "C1", "C2"],
            "ISIN": ["A1", "A2", "A1"],
            "transactionType": ["Buy", "Buy", "Buy"],
            "totalValue": [1000.0, 1500.0, 5000.0],
            "timestamp": [
                sim_start_date
                - pd.DateOffset(months=6),  # Ensure some history before sim start
                sim_start_date - pd.DateOffset(months=4),
                sim_start_date - pd.DateOffset(months=2),
            ],
        }
    )

    env_constructor_params = {
        "customers": mock_customers,
        "assets": mock_assets,
        "prices": mock_prices,
        "transactions": mock_transactions,
        "initial_offset_months": 12,  # Start sim 12 months before latest data point
        "episode_max_steps": 5,
        "accept_probability": 0.9,
        "reward_scale": 10.0,
        "reward_price_weight": 0.5,
        "reward_risk_weight": 0.8,
        "reward_hhi_weight": 1.5,
    }
    try:
        test_env = FinancialRecEnv(**env_constructor_params)
        log.info("Environment instance created successfully for testing.")

        initial_state_vec, initial_info_dict = test_env.reset()
        log.info(
            f"Initial state (C_ID: {initial_info_dict['customer_id']}): shape={initial_state_vec.shape}, first 5 vals={initial_state_vec[:5]}"
        )
        log.info(f"Initial info: {initial_info_dict}")

        for i_step in range(test_env.episode_max_steps):
            random_action = np.random.randint(0, test_env.num_actions)
            next_state_vec, reward_val, term_flag, trunc_flag, step_info_dict = (
                test_env.step(random_action)
            )
            log.info(
                f"Step {i_step + 1}: Action={random_action} (ISIN: {step_info_dict.get('action_isin')}), Reward={reward_val:.3f}, Term={term_flag}, Trunc={trunc_flag}"
            )
            # log.debug(f"  Next State: {next_state_vec[:5]}")
            # log.debug(f"  Step Info: {step_info_dict}")
            if term_flag or trunc_flag:
                log.info("Episode ended.")
                break

    except Exception as e_test:
        log.error(f"Error during FinancialRecEnv module test: {e_test}", exc_info=True)
