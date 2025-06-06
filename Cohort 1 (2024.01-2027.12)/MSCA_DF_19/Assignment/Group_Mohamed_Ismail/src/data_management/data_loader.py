# src/financial_rl_project/data_management/data_loader.py
import pandas as pd
import numpy as np
import logging
import os  # For path joining if needed
from typing import Dict, Optional, Tuple
# from omegaconf import DictConfig # If directly using cfg for paths, but better to pass paths
# import hydra # If using hydra.utils.get_original_cwd()

log = logging.getLogger(__name__)

# Default paths (can be overridden by constructor arguments, which can come from Hydra config)
# These should ideally be relative to the project root or absolute.
# For this example, keeping them as they were. Adjust as needed.
DEFAULT_TRANSACTION_PATH = "../../data/transactions.csv"
DEFAULT_ASSETS_PATH = "../../data/asset_information.csv"
DEFAULT_CLOSE_PRICES_PATH = "../../data/close_prices.csv"
DEFAULT_CUSTOMERS_PATH = "../../data/customer_information.csv"


class DataLoader:
    def __init__(
        self,
        customer_path: str = DEFAULT_CUSTOMERS_PATH,
        asset_path: str = DEFAULT_ASSETS_PATH,
        price_path: str = DEFAULT_CLOSE_PRICES_PATH,
        transaction_path: str = DEFAULT_TRANSACTION_PATH,
    ):
        """
        Initializes the DataLoader with paths to data files.

        Paths can be configured via Hydra by passing them from cfg.data to this constructor.
        Example: DataLoader(customer_path=cfg.data.customers_file_path, ...)
        """
        log.info("Initializing DataLoader...")
        self.paths = {
            "customers": customer_path,
            "assets": asset_path,
            "prices": price_path,
            "transactions": transaction_path,
        }
        log.debug(f"  Customer path: {self.paths['customers']}")
        log.debug(f"  Asset path: {self.paths['assets']}")
        log.debug(f"  Price path: {self.paths['prices']}")
        log.debug(f"  Transaction path: {self.paths['transactions']}")

        self._raw_customers: Optional[pd.DataFrame] = None
        self._raw_assets: Optional[pd.DataFrame] = None
        self._raw_prices: Optional[pd.DataFrame] = None
        self._raw_transactions: Optional[pd.DataFrame] = None
        self._assets_with_volatility: Optional[pd.DataFrame] = None
        self._global_min_time: Optional[pd.Timestamp] = None

    def _resolve_path(self, path: str) -> str:
        """
        Resolves path. If path is relative, it's assumed to be relative
        to the original working directory if run under Hydra, or current CWD otherwise.
        """
        # This logic might be needed if paths in conf are relative to project root
        # and Hydra changes CWD.
        # try:
        #     import hydra
        #     original_cwd = hydra.utils.get_original_cwd()
        #     if not os.path.isabs(path):
        #         return os.path.join(original_cwd, path)
        # except (ImportError, AttributeError): # Not running under Hydra or hydra utils not available
        #     pass # Use path as is
        return path  # For now, assume paths are absolute or correctly relative

    def _load_raw_data(self, force_reload: bool = False) -> None:
        """Loads and performs basic cleaning of all raw dataframes."""
        log.info("--- Loading Raw Data ---")
        data_loaded_this_call = False

        # --- Customers ---
        if self._raw_customers is None or force_reload:
            fpath = self._resolve_path(self.paths["customers"])
            log.info(f"Loading customers from: {fpath}")
            try:
                customers = pd.read_csv(fpath)
                capacity_mask = ~customers["investmentCapacity"].astype(
                    str
                ).str.startswith("Predicted", na=False)
                risk_mask = ~customers["riskLevel"].astype(str).str.startswith(
                    "Predicted", na=False
                )
                self._raw_customers = customers[capacity_mask & risk_mask].copy()
                self._raw_customers["customerID"] = self._raw_customers[
                    "customerID"
                ].astype(str)
                log.info(
                    f" -> Loaded and cleaned {len(self._raw_customers)} customers."
                )
                data_loaded_this_call = True
            except FileNotFoundError:
                log.error(f"Customer file not found: {fpath}")
                raise
            except Exception as e:
                log.error(f"Error loading or cleaning customers: {e}", exc_info=True)
                raise

        # --- Assets ---
        if self._raw_assets is None or force_reload:
            fpath = self._resolve_path(self.paths["assets"])
            log.info(f"Loading assets from: {fpath}")
            try:
                self._raw_assets = pd.read_csv(fpath)
                self._raw_assets["ISIN"] = self._raw_assets["ISIN"].astype(str)
                if "VolatilityQuartile" in self._raw_assets.columns:
                    self._raw_assets["VolatilityQuartile"] = pd.to_numeric(
                        self._raw_assets["VolatilityQuartile"], errors="coerce"
                    )  # Coerce errors to NaN
                log.info(f" -> Loaded {len(self._raw_assets)} assets.")
                data_loaded_this_call = True
            except FileNotFoundError:
                log.error(f"Assets file not found: {fpath}")
                raise
            except Exception as e:
                log.error(f"Error loading assets: {e}", exc_info=True)
                raise

        # --- Transactions ---
        if self._raw_transactions is None or force_reload:
            fpath = self._resolve_path(self.paths["transactions"])
            log.info(f"Loading transactions from: {fpath}")
            try:
                df = pd.read_csv(fpath)
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
                initial_len = len(df)
                df.dropna(subset=["timestamp"], inplace=True)
                if len(df) < initial_len:
                    log.warning(
                        f"Dropped {initial_len - len(df)} transactions with invalid timestamps."
                    )
                df["customerID"] = df["customerID"].astype(str)
                df["ISIN"] = df["ISIN"].astype(str)
                df["totalValue"] = pd.to_numeric(df["totalValue"], errors="coerce")
                # Handle NaNs in totalValue if needed, e.g., fill or drop
                if df["totalValue"].isna().any():
                    log.warning(
                        f"Found {df['totalValue'].isna().sum()} NaN values in 'totalValue'. Consider handling."
                    )
                self._raw_transactions = df
                log.info(
                    f" -> Loaded {len(self._raw_transactions)} transactions (after timestamp cleaning)."
                )
                data_loaded_this_call = True
            except FileNotFoundError:
                log.error(f"Transactions file not found: {fpath}")
                raise
            except Exception as e:
                log.error(f"Error loading transactions: {e}", exc_info=True)
                raise

        # --- Prices ---
        if self._raw_prices is None or force_reload:
            fpath = self._resolve_path(self.paths["prices"])
            log.info(f"Loading prices from: {fpath}")
            try:
                df = pd.read_csv(fpath)
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
                initial_len = len(df)
                df.dropna(subset=["timestamp"], inplace=True)
                if len(df) < initial_len:
                    log.warning(
                        f"Dropped {initial_len - len(df)} price records with invalid timestamps."
                    )
                df["ISIN"] = df["ISIN"].astype(str)
                df["closePrice"] = pd.to_numeric(df["closePrice"], errors="coerce")
                initial_len = len(df)
                df.dropna(subset=["closePrice"], inplace=True)  # Prices are critical
                if len(df) < initial_len:
                    log.warning(
                        f"Dropped {initial_len - len(df)} price records with invalid prices."
                    )
                self._raw_prices = df
                log.info(
                    f" -> Loaded {len(self._raw_prices)} price records (after cleaning)."
                )
                data_loaded_this_call = True
            except FileNotFoundError:
                log.error(f"Prices file not found: {fpath}")
                raise
            except Exception as e:
                log.error(f"Error loading prices: {e}", exc_info=True)
                raise

        if not data_loaded_this_call and not all(
            [
                self._raw_customers is not None,
                self._raw_assets is not None,
                self._raw_transactions is not None,
                self._raw_prices is not None,
            ]
        ):
            log.info("No new data loaded, using cached raw data if available.")

        # Calculate Global Min Time (only if all necessary DFs are loaded)
        if (
            self._raw_transactions is not None
            and not self._raw_transactions.empty
            and self._raw_prices is not None
            and not self._raw_prices.empty
        ):
            min_txn_time = self._raw_transactions["timestamp"].min()
            min_price_time = self._raw_prices["timestamp"].min()
            self._global_min_time = min(min_txn_time, min_price_time)
            log.info(f"Global minimum data timestamp: {self._global_min_time.date()}")
        else:
            log.warning(
                "Transactions or Prices are empty or not loaded. Cannot determine global minimum time."
            )
            self._global_min_time = None  # Explicitly set to None

        log.info("--- Raw Data Loading Finished ---")

    def calculate_static_volatility(
        self, reference_date_str: Optional[str] = None, window_days: int = 30
    ) -> pd.DataFrame:
        """
        Calculates static volatility for assets up to a reference date.
        VolatilityQuartile is added/updated in self._assets_with_volatility.
        """
        if self._raw_assets is None or self._raw_prices is None:
            log.warning(
                "Raw assets or prices not loaded. Attempting to load now for volatility calculation."
            )
            self._load_raw_data()  # Ensure data is available
            if (
                self._raw_assets is None or self._raw_prices is None
            ):  # Check again after load attempt
                log.error(
                    "Cannot calculate volatility: assets or prices still not loaded."
                )
                # Return a copy of raw_assets with NaN volatility if it exists
                return (
                    self._raw_assets.copy().assign(
                        Volatility=np.nan, VolatilityQuartile=np.nan
                    )
                    if self._raw_assets is not None
                    else pd.DataFrame()
                )

        assets_df_copy = self._raw_assets.copy()
        prices_df_copy = self._raw_prices.copy()

        if prices_df_copy.empty:
            log.warning("Price data is empty. Cannot calculate volatility.")
            assets_df_copy["Volatility"] = np.nan
            assets_df_copy["VolatilityQuartile"] = np.nan
            self._assets_with_volatility = assets_df_copy
            return assets_df_copy

        # Determine reference date for volatility calculation
        if reference_date_str:
            try:
                reference_date = pd.to_datetime(reference_date_str).normalize()
            except ValueError:
                log.warning(
                    f"Invalid reference_date_str '{reference_date_str}'. Using latest price timestamp."
                )
                reference_date = prices_df_copy["timestamp"].max().normalize()
        else:
            reference_date = prices_df_copy["timestamp"].max().normalize()

        start_date = reference_date - pd.DateOffset(
            days=window_days - 1
        )  # Window includes reference_date
        log.info(
            f"Calculating static volatility using prices from {start_date.date()} to {reference_date.date()} (window: {window_days} days)."
        )

        prices_in_window = prices_df_copy[
            (prices_df_copy["timestamp"] >= start_date)
            & (prices_df_copy["timestamp"] <= reference_date)
            & (prices_df_copy["ISIN"].isin(assets_df_copy["ISIN"].unique()))
        ].copy()

        if prices_in_window.empty:
            log.warning(
                f"No price data found in the specified window. Volatility will be NaN for all assets."
            )
            assets_df_copy["Volatility"] = np.nan
            assets_df_copy["VolatilityQuartile"] = np.nan
            self._assets_with_volatility = assets_df_copy
            return assets_df_copy

        prices_in_window.sort_values(by=["ISIN", "timestamp"], inplace=True)
        prices_in_window["dailyReturn"] = prices_in_window.groupby("ISIN")[
            "closePrice"
        ].pct_change()

        # Calculate volatility (std dev of daily returns)
        volatility_series = prices_in_window.groupby("ISIN")["dailyReturn"].std(ddof=1)

        # Filter by minimum observations
        min_obs_threshold = max(
            2, min(5, window_days // 2)
        )  # Need at least 2 returns for std dev
        obs_counts = prices_in_window.groupby("ISIN")["dailyReturn"].count()

        # Apply mask: only keep volatility if obs_counts >= min_obs_threshold
        valid_vol_isins = obs_counts[obs_counts >= min_obs_threshold].index
        volatility_series = volatility_series.loc[
            volatility_series.index.isin(valid_vol_isins)
        ]

        volatility_df = volatility_series.reset_index().rename(
            columns={"dailyReturn": "Volatility"}
        )

        assets_df_copy = assets_df_copy.merge(volatility_df, on="ISIN", how="left")

        valid_volatility_values = assets_df_copy["Volatility"].dropna()
        if not valid_volatility_values.empty:
            try:
                # qcut requires at least as many unique values as quantiles. Handle small N.
                num_quantiles = 4
                if len(valid_volatility_values.unique()) < num_quantiles:
                    num_quantiles = max(
                        1, len(valid_volatility_values.unique())
                    )  # reduce quantiles if not enough unique values

                if num_quantiles > 1:
                    quartile_labels = list(range(1, num_quantiles + 1))
                    quartiles = pd.qcut(
                        valid_volatility_values,
                        q=num_quantiles,
                        labels=quartile_labels,
                        duplicates="drop",
                    )
                    assets_df_copy.loc[
                        valid_volatility_values.index, "VolatilityQuartile"
                    ] = quartiles.astype("float")
                elif (
                    num_quantiles == 1
                ):  # Only one unique volatility value (or all same)
                    assets_df_copy.loc[
                        valid_volatility_values.index, "VolatilityQuartile"
                    ] = 1.0  # Assign to first quartile
                else:  # No valid volatility values to cut
                    assets_df_copy["VolatilityQuartile"] = np.nan

            except ValueError as e_qcut:
                log.warning(
                    f"Could not assign {num_quantiles} volatility quartiles: {e_qcut}. Assigning NaN."
                )
                assets_df_copy["VolatilityQuartile"] = np.nan
        else:
            assets_df_copy["VolatilityQuartile"] = (
                np.nan
            )  # No valid volatilities to make quartiles

        # Ensure VolatilityQuartile column exists even if all are NaN
        if "VolatilityQuartile" not in assets_df_copy.columns:
            assets_df_copy["VolatilityQuartile"] = np.nan

        self._assets_with_volatility = assets_df_copy.copy()  # Store the result
        log.info(
            f" -> Calculated static volatility. {len(volatility_df)} assets have non-NaN volatility."
        )
        return self._assets_with_volatility

    def get_train_validation_split(
        self,
        top_n_customers: Optional[int] = 50,
        split_ratio: float = 0.7,
        volatility_reference_date_str: Optional[str] = None,
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
        """
        Loads data, selects customers, calculates volatility based on the training period,
        and splits transactions/prices based on a time-based split_ratio.

        Args:
            top_n_customers: Number of customers with most transactions. If None, all customers.
            split_ratio: Proportion of the data's time duration to use for training.
            volatility_reference_date_str: End date (YYYY-MM-DD) for volatility calc.
                                           If None, uses date corresponding to split point.
        Returns:
            Tuple of (train_data_dict, validation_data_dict).
        """
        self._load_raw_data()  # Ensure raw data is loaded

        if (
            self._raw_transactions is None
            or self._raw_prices is None
            or self._raw_customers is None
            or self._raw_assets is None
        ):
            log.error(
                "One or more raw dataframes are not loaded. Cannot perform split."
            )
            raise RuntimeError("Raw data not available for splitting.")
        if self._global_min_time is None:
            log.error(
                "Global minimum time not determined. Cannot perform time-based split."
            )
            raise RuntimeError("Global minimum time not available for splitting.")

        log.info(
            f"\n--- Preparing Train/Validation Split (Top N Customers: {top_n_customers}, Split Ratio: {split_ratio}) ---"
        )

        # 1. Select Customers
        if self._raw_transactions.empty:
            log.warning(
                "Raw transactions are empty. Cannot select top customers. Using all loaded customers."
            )
            selected_customer_ids = self._raw_customers["customerID"].unique().tolist()
        else:
            customer_counts = self._raw_transactions["customerID"].value_counts()
            if top_n_customers is not None and top_n_customers < len(customer_counts):
                selected_customer_ids = customer_counts.head(
                    top_n_customers
                ).index.tolist()
                log.info(
                    f"Selected top {len(selected_customer_ids)} customers by transaction count."
                )
            else:
                selected_customer_ids = customer_counts.index.tolist()
                log.info(
                    f"Selected all {len(selected_customer_ids)} unique transacting customers."
                )

        customers_for_split = self._raw_customers[
            self._raw_customers["customerID"].isin(selected_customer_ids)
        ].copy()
        if customers_for_split.empty:
            log.error(
                "No customer profiles found for the selected customers. Cannot proceed."
            )
            raise ValueError("No customer data for selected customers.")

        # 2. Determine Time Split Point
        # Use the overall time range from all transactions
        min_data_timestamp = self._raw_transactions["timestamp"].min()
        max_data_timestamp = self._raw_transactions["timestamp"].max()

        if (
            pd.isna(min_data_timestamp)
            or pd.isna(max_data_timestamp)
            or min_data_timestamp >= max_data_timestamp
        ):
            log.warning(
                "Invalid time range in transaction data. Using full data for training, empty for validation."
            )
            split_timestamp = max_data_timestamp
        else:
            total_duration_seconds = (
                max_data_timestamp - min_data_timestamp
            ).total_seconds()
            split_timestamp = min_data_timestamp + pd.Timedelta(
                seconds=total_duration_seconds * split_ratio
            )

        log.info(
            f"Data time range for split: {min_data_timestamp.date()} to {max_data_timestamp.date()}"
        )
        log.info(
            f"Split timestamp: {split_timestamp.date()} (Training data up to and including this date)"
        )

        # 3. Calculate Static Volatility (using data up to the split_timestamp)
        # Use the split_timestamp as the reference for volatility if not overridden
        vol_ref_date_for_calc = (
            volatility_reference_date_str
            if volatility_reference_date_str
            else split_timestamp.strftime("%Y-%m-%d")
        )
        assets_df_with_volatility = self.calculate_static_volatility(
            reference_date_str=vol_ref_date_for_calc
        )
        if assets_df_with_volatility.empty and not self._raw_assets.empty:
            log.warning(
                "Volatility calculation resulted in an empty assets DataFrame. Using raw assets without volatility."
            )
            assets_df_with_volatility = self._raw_assets.copy()
            assets_df_with_volatility["Volatility"] = np.nan
            assets_df_with_volatility["VolatilityQuartile"] = np.nan

        # 4. Split Transactions and Prices based on split_timestamp
        # Filter transactions by selected customers
        transactions_for_split = self._raw_transactions[
            self._raw_transactions["customerID"].isin(selected_customer_ids)
        ].copy()

        train_transactions_df = transactions_for_split[
            transactions_for_split["timestamp"] <= split_timestamp
        ].copy()
        validation_transactions_df = transactions_for_split[
            transactions_for_split["timestamp"] > split_timestamp
        ].copy()

        # Prices are not filtered by customer, but by time. All assets' prices are included.
        train_prices_df = self._raw_prices[
            self._raw_prices["timestamp"] <= split_timestamp
        ].copy()
        validation_prices_df = self._raw_prices[
            self._raw_prices["timestamp"] > split_timestamp
        ].copy()

        # 5. Assemble Data Dictionaries
        train_data = {
            "customers": customers_for_split.copy(),  # Use the selection of customers
            "assets": assets_df_with_volatility.copy(),  # Assets with volatility calculated from train period
            "transactions": train_transactions_df,
            "prices": train_prices_df,
        }
        validation_data = {
            "customers": customers_for_split.copy(),  # Same customers for validation
            "assets": assets_df_with_volatility.copy(),  # Same assets/volatility definition
            "transactions": validation_transactions_df,
            "prices": validation_prices_df,
        }
        log.info("--- Data Split Summary ---")
        for name, data_dict in [
            ("Training", train_data),
            ("Validation", validation_data),
        ]:
            log.info(f" {name} Set:")
            for key, df_val in data_dict.items():
                log.info(f"  {key.capitalize()}: {df_val.shape[0]} rows")
            if not data_dict["transactions"].empty:
                log.info(
                    f"    Transactions time range: {data_dict['transactions']['timestamp'].min().date()} to {data_dict['transactions']['timestamp'].max().date()}"
                )
            if not data_dict["prices"].empty:
                log.info(
                    f"    Prices time range: {data_dict['prices']['timestamp'].min().date()} to {data_dict['prices']['timestamp'].max().date()}"
                )

        return train_data, validation_data


# For standalone testing of DataLoader
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
    )
    log.info("--- Running data_loader.py script directly for testing ---")

    # Important: For this test to work with real data,
    # ensure DEFAULT paths are correct or provide paths here.
    # Example:
    # loader = DataLoader(
    #     customer_path="path/to/your/customers.csv", # etc.
    # )
    try:
        loader = DataLoader()  # Uses default paths defined in the class

        # Test loading raw data
        # loader._load_raw_data(force_reload=True)
        # if loader._raw_prices is not None:
        #     log.info(f"Raw prices head:\n{loader._raw_prices.head()}")
        # else:
        #     log.warning("Raw prices not loaded.")

        # Test volatility calculation
        # assets_vol = loader.calculate_static_volatility(reference_date_str="2022-01-01")
        # log.info(f"Assets with volatility head:\n{assets_vol.head()}")

        # Test train/validation split
        train_set, val_set = loader.get_train_validation_split(
            top_n_customers=20,  # Smaller number for quicker test
            split_ratio=0.8,
        )
        log.info("Train/Validation split successful.")
        log.info(f"Training transactions: {len(train_set['transactions'])}")
        log.info(f"Validation transactions: {len(val_set['transactions'])}")

    except FileNotFoundError as fnf_err:
        log.error(
            f"TEST FAILED: A data file was not found. Please check default paths or provide correct paths. Error: {fnf_err}"
        )
    except Exception as main_err:
        log.error(f"Error during DataLoader test: {main_err}", exc_info=True)
