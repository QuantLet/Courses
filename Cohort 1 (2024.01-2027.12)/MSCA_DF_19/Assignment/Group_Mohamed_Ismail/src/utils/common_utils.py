# src/utils/common_utils.py
import numpy as np
import pandas as pd  # For type hinting if data_dict values are DataFrames
import random
import torch
import logging
import os
import sys  # For sys.exit
from omegaconf import DictConfig, OmegaConf, open_dict  # For modifying cfg
from typing import Tuple, Dict, Any, TYPE_CHECKING

# For type hinting without circular imports at module load time
if TYPE_CHECKING:
    from environment.financial_env import FinancialRecEnv  # Assuming flat src structure
    from data_management.data_loader import DataLoader
    from data_management.mock_data import create_mock_data

log = logging.getLogger(__name__)

# --- Optional Dependency Checks ---
CVXPY_OK = False
try:
    import cvxpy  # type: ignore

    CVXPY_OK = True
    log.debug("CVXPY imported successfully.")
except ImportError:
    log.info(
        "CVXPY not installed. MVO heuristic will fall back to Equal Weight if selected."
    )

PLOTLY_OK = False
try:
    import plotly  # type: ignore

    PLOTLY_OK = True
    log.debug("Plotly imported successfully.")
except ImportError:
    log.info("Plotly not installed. Optuna visualizations will not be available.")


# --- Core Utility Functions ---
def set_seeds(seed: int) -> None:
    """Sets random seeds for reproducibility across relevant libraries."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
        # Ensuring deterministic behavior can impact performance.
        # Set benchmark to False if exact reproducibility is critical.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    log.info(f"Global random seeds set to {seed}.")


def get_torch_device(cfg_device_name: str) -> torch.device:
    """
    Determines the torch.device based on configuration and availability.

    Args:
        cfg_device_name: Device string from config (e.g., "auto", "cuda", "cpu").

    Returns:
        A torch.device object.
    """
    if cfg_device_name.lower() == "auto":
        resolved_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif cfg_device_name.lower() == "cuda":
        if not torch.cuda.is_available():
            log.warning(
                "CUDA was configured, but is not available. Falling back to CPU."
            )
            resolved_device = torch.device("cpu")
        else:
            resolved_device = torch.device("cuda")
    elif cfg_device_name.lower() == "cpu":
        resolved_device = torch.device("cpu")
    else:
        log.warning(
            f"Unknown device '{cfg_device_name}' specified in config. Defaulting to CPU."
        )
        resolved_device = torch.device("cpu")
    log.info(f"Selected PyTorch device: {resolved_device}")
    return resolved_device


def setup_data_and_environment(
    cfg: DictConfig,
) -> Tuple["FinancialRecEnv", Dict[str, pd.DataFrame], torch.device]:
    """
    Loads data (real or mock) using DataLoader or mock_data generator,
    and initializes the FinancialRecEnv.
    Updates the input cfg object with derived parameters like state_size, action_size.

    Args:
        cfg: The main Hydra configuration object. Expected to have sub-configs
             like cfg.data, cfg.environment, cfg.use_real_data.

    Returns:
        A tuple containing:
            - env (FinancialRecEnv): The initialized environment instance.
            - env_data_dict (Dict[str, pd.DataFrame]): Dictionary of DataFrames
              (customers, assets, prices, transactions) used to init the env.
              Useful for heuristics or direct data inspection.
            - device (torch.device): The torch device determined for the run.
    """
    # --- Import necessary modules dynamically to avoid import cycles if this utils is imported early ---
    # And to ensure imports are relative to the 'src' directory (or wherever packages are found)
    from data_management.data_loader import DataLoader
    from data_management.mock_data import create_mock_data
    from environment.financial_env import FinancialRecEnv

    log.info("--- Setting up Data Source and Financial Environment ---")

    # Determine torch device (already done in run.py, but can be self-contained here too)
    # If cfg.device_actual is already set by run.py, use it. Otherwise, determine it.
    if hasattr(cfg, "device_actual") and cfg.device_actual:
        device = torch.device(cfg.device_actual)  # Convert string back to device object
    else:
        device = get_torch_device(cfg.device)  # cfg.device from main config.yaml
        with open_dict(cfg):  # Allow adding new keys to cfg
            cfg.device_actual = str(device)  # Store the actual device string in cfg

    # --- 1. Load Data ---
    env_data_dict: Dict[str, pd.DataFrame] = {}
    if cfg.use_real_data:
        log.info("Attempting to load REAL data using DataLoader...")
        try:
            # Construct paths for DataLoader.
            # Paths in cfg.data.*_file_path are expected to be absolute
            # or resolvable from the original CWD if Hydra changes it.
            # DataLoader._resolve_path (if implemented with hydra.utils.get_original_cwd())
            # would handle paths relative to project root.
            # For simplicity, assume paths in cfg are made absolute or DataLoader handles them.
            loader = DataLoader(
                customer_path=cfg.data.get("customers_file_path", None),
                asset_path=cfg.data.get("assets_file_path", None),
                price_path=cfg.data.get("prices_file_path", None),
                transaction_path=cfg.data.get("transactions_file_path", None),
            )
            # For setting up the main environment, we might not always need a train/validation split of the data.
            # Often, the environment uses the "training" portion or all available historical data.
            # Let's assume get_train_validation_split with ratio 1.0 gives us all data structured.
            # Or, DataLoader could have a get_full_processed_data() method.
            log.info(
                f"DataLoader configured with top_n_customers: {cfg.data.top_n_customers}, split_ratio for this load: 1.0 (full dataset for env)"
            )
            # We use a split_ratio of 1.0 here to get all data for the environment setup.
            # If separate train/eval environments are needed with distinct data,
            # this function would need to be called differently or manage splits internally.
            env_data_dict, _ = loader.get_train_validation_split(
                top_n_customers=cfg.data.top_n_customers,
                split_ratio=1.0,  # Get all data processed for environment setup
            )
            log.info("Real data loaded and processed via DataLoader.")
            if env_data_dict["transactions"].empty or env_data_dict["prices"].empty:
                log.error(
                    "Loaded real data has empty transactions or prices after DataLoader processing."
                )
                raise ValueError(
                    "Real data processing resulted in empty critical DataFrames."
                )
        except FileNotFoundError as e:
            log.error(
                f"FATAL: Real data file not found. Check paths in conf/data.yaml or DataLoader defaults. Error: {e}"
            )
            sys.exit(1)
        except Exception as e:
            log.error(
                f"FATAL: Error loading or processing real data: {e}", exc_info=True
            )
            sys.exit(1)
    else:
        log.info("Using MOCK data generation...")
        try:
            customers_df, assets_df, prices_df, transactions_df = create_mock_data(
                seed=cfg.data.mock_data_seed,  # From conf/data.yaml
                n_assets=cfg.data.mock_n_assets,
                n_users=cfg.data.mock_n_users,
                price_start=cfg.data.mock_price_start,
                total_days=cfg.data.mock_total_days,
                annual_mu_range=tuple(cfg.data.mock_annual_mu_range),  # Ensure tuple
                annual_vol_range=tuple(cfg.data.mock_annual_vol_range),  # Ensure tuple
                daily_trading_days=cfg.data.mock_daily_trading_days,
                dataset_type=cfg.data.mock_dataset_type,
            )
            env_data_dict = {
                "customers": customers_df,
                "assets": assets_df,
                "prices": prices_df,
                "transactions": transactions_df,
            }
            log.info("Mock data generated successfully.")
        except Exception as e:
            log.error(f"FATAL: Error generating mock data: {e}", exc_info=True)
            sys.exit(1)

    # --- 2. Initialize Environment ---
    log.info("Initializing FinancialRecEnv with loaded/generated data...")
    env_constructor_params = {
        "customers": env_data_dict["customers"],
        "assets": env_data_dict["assets"],
        "prices": env_data_dict["prices"],
        "transactions": env_data_dict["transactions"],
        # Parameters from cfg.environment (conf/environment.yaml)
        "initial_offset_months": cfg.environment.initial_offset_months,
        "episode_max_steps": cfg.environment.episode_max_steps,
        "accept_probability": cfg.environment.accept_probability,
        "reward_scale": cfg.environment.reward_scale,
        "reward_price_weight": cfg.environment.reward_price_weight,
        "reward_risk_weight": cfg.environment.reward_risk_weight,
        "reward_hhi_weight": cfg.environment.reward_hhi_weight,
    }

    try:
        env = FinancialRecEnv(**env_constructor_params)

        # Store derived environment properties in the global cfg for access elsewhere
        with open_dict(cfg):  # Allows adding new keys to the Hydra config object
            cfg.state_size = env.state_dimension
            cfg.action_size = env.num_actions
            cfg.asset_names = env.asset_isins_list  # List of ISINs used as actions
            cfg.num_assets = len(env.asset_isins_list)

        log.info(
            f"FinancialRecEnv initialized. State_size: {cfg.state_size}, Action_size: {cfg.action_size}"
        )
        return env, env_data_dict, device
    except Exception as e:
        log.error(f"FATAL: Error initializing FinancialRecEnv: {e}", exc_info=True)
        sys.exit(1)
