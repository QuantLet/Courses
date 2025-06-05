# src/run.py
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
import logging
import os
import pandas as pd
import sys
import numpy as np
from tqdm import tqdm
import random
import json
import optuna  # For Optuna exceptions and type hinting
from typing import Optional

# --- Import project modules ---
from utils.common_utils import (
    set_seeds,
    get_torch_device,
    setup_data_and_environment,
    CVXPY_OK,
    PLOTLY_OK,
)
from agent.dqn_agent import DQNAgent
from agent.ddqn_agent import DDQNAgent
from agent.a2c_agent import A2CAgent
from training.core_train_loops import train_dqn_episode, train_a2c_episode
from training.optuna_setup import (
    create_optuna_study,
    objective_function,
)
from evaluation.evaluator import evaluate_policy



from utils.plotting import ( 
    plot_training_results,
    plot_evaluation_comparison,
    plot_optuna_visualizations,  
)

# Heuristics still a placeholder
log_heuristic = logging.getLogger("heuristic_placeholders")


def get_heuristic_policy_decision_func(strategy_name: str, **kwargs):
    log_heuristic.warning(
        f"HEURISTIC_PLACEHOLDER for {strategy_name}. Using random policy."
    )
    action_size = kwargs.get("num_assets", 1)
    if action_size == 0:
        action_size = 1
    return lambda state, **current_call_kwargs: random.choice(np.arange(action_size))


log = logging.getLogger(__name__)


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main_app(cfg: DictConfig) -> None:
    # ... (output_dir, initial logging, setup, data/env loading - all same) ...
    try:
        output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    except Exception:
        log.warning("Could not get output_dir via HydraConfig. Using os.getcwd().")
        output_dir = os.getcwd()

    log.info(f"Effective Hydra output directory: {output_dir}")
    log.info(
        f"Run Config: Seed={cfg.seed}, Device={cfg.device}, UseRealData={cfg.use_real_data}, AgentType={cfg.agent.type}"
    )
    log.debug(f"Full configuration:\n{OmegaConf.to_yaml(cfg)}")

    set_seeds(cfg.seed)
    device = get_torch_device(cfg.device)
    with open_dict(cfg):
        cfg.device_actual = str(device)

    try:
        env, env_data_dict, _ = setup_data_and_environment(cfg)
        prices_df_for_eval = env_data_dict["prices"]
    except Exception as e:
        log.error(f"Failed during data/env setup: {e}", exc_info=True)
        sys.exit(1)

    all_eval_results = {}


    # --- 3. Optuna Hyperparameter Optimization ---
    best_hyperparams_from_optuna = {}
    optuna_study: Optional[optuna.Study] = None  # Initialize study object to None

    if cfg.run_optuna:
        log.info(
            f"\n--- Starting Optuna Study ({cfg.optuna.n_trials} trials) for {cfg.agent.type.upper()} ---"
        )
        if cfg.agent.type.upper() not in ["DQN", "DDQN"]:
            log.warning(
                f"Optuna objective currently designed for DQN/DDQN. Tuning for {cfg.agent.type.upper()} might need objective adjustments."
            )
        try:
            optuna_study = create_optuna_study(
                cfg_optuna=cfg.optuna, output_dir=output_dir
            )
            optuna_study.optimize(
                lambda trial: objective_function(
                    trial,
                    cfg_fixed=cfg,
                    env_data_dict_for_trial=env_data_dict,
                    device=device,
                ),
                n_trials=cfg.optuna.n_trials,
                timeout=cfg.optuna.timeout_seconds
                if cfg.optuna.timeout_seconds
                else None,
                gc_after_trial=True,
            )
            if optuna_study.best_trial:
                best_hyperparams_from_optuna = optuna_study.best_trial.params
                best_value = optuna_study.best_trial.value
                log.info(f"Optuna study finished. Best trial value: {best_value:.4f}")
                log.info(f"Best hyperparameters: {best_hyperparams_from_optuna}")
                params_save_path = os.path.join(
                    output_dir, cfg.best_params_save_filename
                )
                with open(params_save_path, "w") as f:
                    json.dump(best_hyperparams_from_optuna, f, indent=4)
                log.info(f"Best Optuna hyperparameters saved to {params_save_path}")
            else:
                log.warning("Optuna study completed but no best trial found.")
        except Exception as e_optuna:
            log.error(f"Error during Optuna optimization: {e_optuna}", exc_info=True)
            log.warning("Proceeding with default agent hyperparameters.")
    else:
        log.info("Optuna optimization disabled by config.")


    # --- 4. Initialize and Train Final Agent ---
    training_results = {}
    if cfg.train_final_agent:

        log.info(
            f"\n--- Initializing Agent (Type: {cfg.agent.type.upper()}) for Final Training ---"
        )
        agent_cfg_resolved_dict = OmegaConf.to_container(cfg.agent, resolve=True)
        if best_hyperparams_from_optuna:
            log.info(
                f"Updating agent config with Optuna best params: {best_hyperparams_from_optuna}"
            )
            for key, value in best_hyperparams_from_optuna.items():
                if key in agent_cfg_resolved_dict or key == "hidden_layers":
                    log.debug(
                        f"Optuna overriding/setting '{key}': {agent_cfg_resolved_dict.get(key)} -> {value}"
                    )
                    agent_cfg_resolved_dict[key] = value
                elif key not in ["n_layers", "layer1_size"]:
                    log.warning(
                        f"Optuna suggested param '{key}' not directly in base agent config, adding it."
                    )
                    agent_cfg_resolved_dict[key] = value
            if (
                "hidden_layers" not in best_hyperparams_from_optuna
                and "n_layers" in best_hyperparams_from_optuna
                and "layer1_size" in best_hyperparams_from_optuna
            ):
                n_l, l1_s = (
                    best_hyperparams_from_optuna["n_layers"],
                    best_hyperparams_from_optuna["layer1_size"],
                )
                h_layers = [l1_s] + [
                    max(32, h_layers[-1] // 2)
                    for _ in range(n_l - 1)
                    for h_layers in [[l1_s]]
                ]  # Simplified construction
                agent_cfg_resolved_dict["hidden_layers"] = tuple(h_layers)
                log.info(
                    f"Reconstructed hidden_layers for final agent: {tuple(h_layers)}"
                )
        final_agent_hydra_cfg = OmegaConf.create(agent_cfg_resolved_dict)
        agent_type_upper = cfg.agent.type.upper()
        try:
            if agent_type_upper == "DQN":
                agent_instance = DQNAgent(
                    cfg.state_size,
                    cfg.action_size,
                    final_agent_hydra_cfg,
                    cfg.seed,
                    device,
                )
            elif agent_type_upper == "DDQN":
                agent_instance = DDQNAgent(
                    cfg.state_size,
                    cfg.action_size,
                    final_agent_hydra_cfg,
                    cfg.seed,
                    device,
                )
            elif agent_type_upper == "A2C":
                agent_instance = A2CAgent(
                    cfg.state_size,
                    cfg.action_size,
                    final_agent_hydra_cfg,
                    cfg.seed,
                    device,
                )
            else:
                log.error(f"Unknown agent type: {cfg.agent.type}")
                sys.exit(1)
        except Exception as e:
            log.error(
                f"Failed to initialize {agent_type_upper} agent: {e}", exc_info=True
            )
            sys.exit(1)
        log.info(
            f"--- Starting {agent_type_upper} Training for {cfg.training.num_episodes} episodes ---"
        )
        rewards_log, lengths_log, losses1_log, losses2_log, eps_log = [], [], [], [], []
        if agent_type_upper in ["DQN", "DDQN"]:
            epsilon = cfg.training.epsilon_start
        for ep_num in tqdm(
            range(1, cfg.training.num_episodes + 1), desc=f"{agent_type_upper} Training"
        ):
            if agent_type_upper in ["DQN", "DDQN"]:
                ep_r, ep_s, ep_l1 = train_dqn_episode(env, agent_instance, epsilon)
                losses1_log.append(ep_l1 if ep_l1 is not None else np.nan)
                epsilon = max(
                    cfg.training.epsilon_end, epsilon * cfg.training.epsilon_decay
                )
                eps_log.append(epsilon)
                loss_str = f"{agent_type_upper}_Loss: {ep_l1 if ep_l1 is not None else float('nan'):.4f}"
            elif agent_type_upper == "A2C":
                ep_r, ep_s, ep_l1, ep_l2 = train_a2c_episode(env, agent_instance)
                losses1_log.append(ep_l1 if ep_l1 is not None else np.nan)
                losses2_log.append(ep_l2 if ep_l2 is not None else np.nan)
                loss_str = f"ActorL: {ep_l1 if ep_l1 is not None else float('nan'):.4f}, CriticL: {ep_l2 if ep_l2 is not None else float('nan'):.4f}"
            rewards_log.append(ep_r)
            lengths_log.append(ep_s)
            if ep_num % 100 == 0 or ep_num == cfg.training.num_episodes:
                avg_r_100 = np.mean(rewards_log[-100:]) if rewards_log else 0.0
                eps_str = (
                    f"Eps: {epsilon:.4f}"
                    if agent_type_upper in ["DQN", "DDQN"]
                    else "PolicyStochastic"
                )
                log.info(
                    f"Ep {ep_num}/{cfg.training.num_episodes} | AvgR(100): {avg_r_100:.2f} | {eps_str} | {loss_str}"
                )
        log.info(f"--- {agent_type_upper} Training Finished ---")
        model_save_path_or_prefix = (
            os.path.join(output_dir, cfg.dqn_save_filename)
            if agent_type_upper in ["DQN", "DDQN"]
            else os.path.join(output_dir, f"{agent_type_upper.lower()}_agent_final")
        )
        if hasattr(agent_instance, "save_model"):
            agent_instance.save_model(model_save_path_or_prefix)
        else:
            log.error(f"Agent {agent_type_upper} has no save_model method.")
        training_results = {"rewards": rewards_log, "lengths": lengths_log}
        if agent_type_upper in ["DQN", "DDQN"]:
            training_results["loss"], training_results["epsilon"] = losses1_log, eps_log
        elif agent_type_upper == "A2C":
            training_results["actor_loss"], training_results["critic_loss"] = (
                losses1_log,
                losses2_log,
            )

        summary_df = pd.DataFrame(training_results)  
        summary_path = os.path.join(
            output_dir, f"{agent_type_upper.lower()}_training_summary.csv"
        )
        summary_df.to_csv(summary_path, index=False)
        log.info(f"{agent_type_upper} training summary saved to {summary_path}")
    else:  # Not training final agent
        log.info("Skipping agent training as per 'train_final_agent: false'.")
        if cfg.evaluate_final_agent:
            pass

    # --- 5. Evaluate Trained Agent ---
    if cfg.evaluate_final_agent and agent_instance is not None:

        agent_type_upper_eval = cfg.agent.type.upper()
        log.info(f"\n--- Evaluating Trained {agent_type_upper_eval} Agent ---")
        eval_env, _, _ = setup_data_and_environment(cfg)
        eval_specific_kwargs = {}
        if agent_type_upper_eval in ["DQN", "DDQN"]:
            eval_specific_kwargs["epsilon"] = 0.0
        elif (
            agent_type_upper_eval == "A2C"
            and hasattr(agent_instance, "choose_action")
            and "store_for_trajectory"
            in agent_instance.choose_action.__code__.co_varnames
        ):
            eval_specific_kwargs["store_for_trajectory"] = False
        (
            agent_avg_r,
            agent_std_r,
            agent_avg_l,
            agent_all_rewards,
            agent_all_lengths,
            agent_detailed_logs,
        ) = evaluate_policy(
            agent_instance.choose_action,
            eval_env,
            cfg.evaluation.num_eval_episodes,
            True,
            f"{agent_type_upper_eval} (Trained)",
            log_detailed_episodes=cfg.evaluation.get("log_n_detailed_episodes", 0),
            **eval_specific_kwargs,
        )
        all_eval_results[f"{agent_type_upper_eval} (Trained)"] = {
            "avg_reward": agent_avg_r,
            "std_reward": agent_std_r,
            "avg_length": agent_avg_l,
            "rewards": agent_all_rewards,
            "lengths": agent_all_lengths,
            "detailed_logs": agent_detailed_logs,
        }
        log.info(
            f"{agent_type_upper_eval} (Trained): Avg Reward={agent_avg_r:.2f} +/- {agent_std_r:.2f}"
        )

    elif cfg.evaluate_final_agent:
        log.warning("Evaluation of final agent requested, but agent is not available.")

    # --- 6. Generate Plots ---
    log.info("\n--- Generating Plots ---")  # Changed from placeholder message
    if training_results:
        plot_training_results(
            results_dict=training_results,  # Use the populated training_results
            cfg_training_params=cfg.training,  # Pass relevant training config
            save_dir=output_dir,
            policy_name_prefix=f"{cfg.agent.type.upper()}_Final",
        )

    # Ensure optuna_study is the actual study object if Optuna ran
    if (
        cfg.run_optuna and optuna_study and PLOTLY_OK
    ):  
        plot_optuna_visualizations(
            study=optuna_study,  
            save_dir=output_dir,
        )
    elif cfg.run_optuna and not PLOTLY_OK:
        log.warning(
            "Optuna ran, but Plotly is not available for visualizations. Skipping Optuna plots."
        )

    if all_eval_results:  
        plot_evaluation_comparison(
            all_eval_results=all_eval_results,
            num_eval_episodes_for_title=cfg.evaluation.num_eval_episodes,  # For plot title
            save_dir=output_dir,
        )

    # --- 7. Print Summary ---
    log.info("\n--- Run Summary ---")
    log.info(f"Agent Type: {cfg.agent.type.upper()}")
    if all_eval_results:
        log.info(
            f"\nPolicy Evaluation Results ({cfg.evaluation.num_eval_episodes} episodes each):"
        )
        header = f"{'Policy':<25} | {'Avg Reward':<12} | {'Std Reward':<12} | {'Avg Length':<12}"
        log.info(header)
        log.info("-" * len(header))
        sorted_policies = sorted(
            all_eval_results.items(),
            key=lambda item: item[1]["avg_reward"],
            reverse=True,
        )
        for name, res in sorted_policies:
            log.info(
                f"{name:<25} | {res['avg_reward']:<12.2f} | {res.get('std_reward', float('nan')):<12.2f} | {res.get('avg_length', float('nan')):<12.1f}"
            )
        log.info("-" * len(header))

    log.info(f"\nRun finished. Outputs in: {output_dir}")
    log.info("--- Financial RL Project Script Completed ---")


if __name__ == "__main__":
    main_app()
