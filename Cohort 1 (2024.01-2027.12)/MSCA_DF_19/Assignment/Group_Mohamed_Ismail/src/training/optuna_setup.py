# src/training/optuna_setup.py
import optuna
import logging
import os
import torch
from omegaconf import DictConfig, OmegaConf
from typing import Dict, Any, Tuple
import pandas as pd


from agent.dqn_agent import DQNAgent
from agent.ddqn_agent import DDQNAgent
from agent.a2c_agent import A2CAgent 
from environment.financial_env import FinancialRecEnv
from training.core_train_loops import train_dqn_episode, train_a2c_episode 
from evaluation.evaluator import evaluate_policy

log = logging.getLogger(__name__)

def create_optuna_study(cfg_optuna: DictConfig, output_dir: str) -> optuna.Study:

    study_name = cfg_optuna.study_name
    storage_name = cfg_optuna.storage_db_name
    storage_path = f"sqlite:///{os.path.join(output_dir, storage_name)}"
    pruner_cfg = cfg_optuna.get('pruner', None)
    pruner_instance = None
    if pruner_cfg and pruner_cfg.get('_target_'):
        try:
            if pruner_cfg._target_ == "optuna.pruners.MedianPruner":
                pruner_instance = optuna.pruners.MedianPruner(
                    n_startup_trials=pruner_cfg.get("n_startup_trials", 5),
                    n_warmup_steps=pruner_cfg.get("n_warmup_steps", 0),
                    interval_steps=pruner_cfg.get("interval_steps", 1)
                )
            elif pruner_cfg._target_ == "optuna.pruners.NopPruner":
                pruner_instance = optuna.pruners.NopPruner()
            else: log.warning(f"Unsupported pruner: {pruner_cfg._target_}. Using NopPruner."); pruner_instance = optuna.pruners.NopPruner()
        except Exception as e: log.error(f"Error instantiating pruner: {e}. Using NopPruner."); pruner_instance = optuna.pruners.NopPruner()
    else: log.info("No pruner configured. Using NopPruner."); pruner_instance = optuna.pruners.NopPruner()
    study = optuna.create_study(study_name=study_name, storage=storage_path, load_if_exists=True, direction="maximize", pruner=pruner_instance)
    log.info(f"Optuna study '{study_name}' from '{storage_path}'. Finished trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
    return study


def objective_function(
    trial: optuna.Trial,
    cfg_fixed: DictConfig,
    env_data_dict_for_trial: Dict[str, pd.DataFrame],
    device: torch.device
) -> float:
    log.info(f"\n--- Optuna Trial {trial.number} Starting for Agent Type: {cfg_fixed.agent.type.upper()} ---")
    
    agent_type_upper = cfg_fixed.agent.type.upper()
    trial_agent_cfg_dict = OmegaConf.to_container(cfg_fixed.agent, resolve=True) 

    # --- 1. Suggest Hyperparameters (Agent-Specific) ---
    if agent_type_upper in ["DQN", "DDQN"]:
        trial_agent_cfg_dict["lr"] = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        trial_agent_cfg_dict["gamma"] = trial.suggest_float("gamma", 0.95, 0.999)
        trial_agent_cfg_dict["batch_size"] = trial.suggest_categorical("batch_size", [32, 64, 128])
        trial_agent_cfg_dict["tau"] = trial.suggest_float("tau", 1e-4, 1e-2, log=True)
        
        n_layers = trial.suggest_int("n_layers_dqn", 1, 3) # Suffix to avoid clash if A2C has different layer suggestions
        layer1_size = trial.suggest_categorical("layer1_size_dqn", [64, 128, 256])
        hidden_layers_list = [layer1_size]
        current_size = layer1_size
        for _ in range(n_layers - 1):
            next_size = max(32, current_size // 2)
            if next_size == current_size and current_size <= 64 and n_layers > 1: hidden_layers_list.append(next_size); break
            hidden_layers_list.append(next_size); current_size = next_size
        trial_agent_cfg_dict["hidden_layers"] = tuple(h for i, h in enumerate(hidden_layers_list) if h > 0 and (i == 0 or h != hidden_layers_list[i-1] or hidden_layers_list[i-1] > 64 ))
        log.debug(f"Trial {trial.number} [DQN/DDQN] suggested hidden_layers: {trial_agent_cfg_dict['hidden_layers']}")

    elif agent_type_upper == "A2C":
        trial_agent_cfg_dict["actor_lr"] = trial.suggest_float("actor_lr", 1e-5, 1e-3, log=True)
        trial_agent_cfg_dict["critic_lr"] = trial.suggest_float("critic_lr", 1e-5, 1e-3, log=True) 
        trial_agent_cfg_dict["gamma"] = trial.suggest_float("gamma_a2c", 0.95, 0.999) # 
        trial_agent_cfg_dict["entropy_coefficient"] = trial.suggest_float("entropy_coefficient", 0.001, 0.05, log=True)
        
        n_actor_layers = trial.suggest_int("n_actor_layers", 1, 2)
        actor_l1_size = trial.suggest_categorical("actor_l1_size", [64, 128])
        actor_h_list = [actor_l1_size] + [max(32, actor_l1_size//(2**i)) for i in range(1, n_actor_layers)]
        trial_agent_cfg_dict["actor_hidden_layers"] = tuple(actor_h_list)
        log.debug(f"Trial {trial.number} [A2C] suggested actor_hidden_layers: {trial_agent_cfg_dict['actor_hidden_layers']}")

        n_critic_layers = trial.suggest_int("n_critic_layers", 1, 2)
        critic_l1_size = trial.suggest_categorical("critic_l1_size", [64, 128])
        critic_h_list = [critic_l1_size] + [max(32, critic_l1_size//(2**i)) for i in range(1, n_critic_layers)]
        trial_agent_cfg_dict["critic_hidden_layers"] = tuple(critic_h_list)
        log.debug(f"Trial {trial.number} [A2C] suggested critic_hidden_layers: {trial_agent_cfg_dict['critic_hidden_layers']}")
    else:
        log.error(f"Trial {trial.number}: Optuna objective not configured for agent type '{agent_type_upper}'. Pruning.")
        raise optuna.exceptions.TrialPruned()

    trial_agent_hydra_cfg = OmegaConf.create(trial_agent_cfg_dict)
    log.debug(f"Trial {trial.number} Effective Agent Config: {OmegaConf.to_yaml(trial_agent_hydra_cfg)}")

    # --- 2. Setup Environment for this Trial ---
    env_params_for_trial = {**env_data_dict_for_trial, **OmegaConf.to_container(cfg_fixed.environment, resolve=True)}
    try:
        trial_env = FinancialRecEnv(**env_params_for_trial)
    except Exception as e: log.error(f"Trial {trial.number}: Failed to create env: {e}", exc_info=True); raise

    # --- 3. Initialize Agent for this Trial ---
    trial_agent = None
    try:
        if agent_type_upper == "DQN":
            trial_agent = DQNAgent(cfg_fixed.state_size, cfg_fixed.action_size, trial_agent_hydra_cfg, cfg_fixed.seed, device)
        elif agent_type_upper == "DDQN":
            trial_agent = DDQNAgent(cfg_fixed.state_size, cfg_fixed.action_size, trial_agent_hydra_cfg, cfg_fixed.seed, device)
        elif agent_type_upper == "A2C":
            trial_agent = A2CAgent(cfg_fixed.state_size, cfg_fixed.action_size, trial_agent_hydra_cfg, cfg_fixed.seed, device)
    except Exception as e: log.error(f"Trial {trial.number}: Failed to init agent {agent_type_upper}: {e}", exc_info=True); raise

    # --- 4. Short Training Phase for the Trial Agent ---
    num_train_eps_trial = cfg_fixed.optuna.training_episodes_per_trial
    if agent_type_upper in ["DQN", "DDQN"]:
        epsilon = cfg_fixed.training.epsilon_start
        epsilon_end_trial = cfg_fixed.training.epsilon_end
        epsilon_decay_trial = cfg_fixed.training.epsilon_decay
    
    log.info(f"Trial {trial.number}: Training {agent_type_upper} for {num_train_eps_trial} episodes.")
    for ep in range(1, num_train_eps_trial + 1):
        if agent_type_upper in ["DQN", "DDQN"]:
            train_dqn_episode(trial_env, trial_agent, epsilon)
            epsilon = max(epsilon_end_trial, epsilon * epsilon_decay_trial)
        elif agent_type_upper == "A2C":
            train_a2c_episode(trial_env, trial_agent) # A2C handles exploration via policy stochasticity
        
        eval_interval = max(1, num_train_eps_trial // cfg_fixed.optuna.get("pruning_checks_per_trial", 5))
        if ep % eval_interval == 0 or ep == num_train_eps_trial:
            log.debug(f"Trial {trial.number}, Ep {ep}: Intermediate evaluation...")
            eval_kwargs = {"epsilon": 0.0} if agent_type_upper in ["DQN", "DDQN"] else {}
            if agent_type_upper == "A2C" and hasattr(trial_agent, 'choose_action') and \
               'store_for_trajectory' in trial_agent.choose_action.__code__.co_varnames:
                eval_kwargs["store_for_trajectory"] = False
            
            avg_intermediate_r, _, _, _, _, _ = evaluate_policy(
                trial_agent.choose_action, trial_env, cfg_fixed.optuna.evaluation_episodes_per_trial,
                True, f"Trial {trial.number} Mid-Eval Ep {ep}", log_detailed_episodes=0, **eval_kwargs)
            trial.report(avg_intermediate_r, ep)
            if trial.should_prune():
                log.info(f"Trial {trial.number} pruned at ep {ep} with reward {avg_intermediate_r:.3f}.")
                raise optuna.exceptions.TrialPruned()
    
    # --- 5. Final Evaluation for this Trial's Agent ---
    log.info(f"Trial {trial.number}: Final evaluation...")
    final_eval_kwargs = {"epsilon": 0.0} if agent_type_upper in ["DQN", "DDQN"] else {}
    if agent_type_upper == "A2C" and hasattr(trial_agent, 'choose_action') and \
        'store_for_trajectory' in trial_agent.choose_action.__code__.co_varnames:
        final_eval_kwargs["store_for_trajectory"] = False

    avg_final_eval_reward, _, _, _, _, _ = evaluate_policy(
        trial_agent.choose_action, trial_env, cfg_fixed.optuna.evaluation_episodes_per_trial,
        True, f"Trial {trial.number} Final Eval", log_detailed_episodes=0, **final_eval_kwargs)
    
    log.info(f"Trial {trial.number} Finished. Final Avg Eval Reward: {avg_final_eval_reward:.4f}")
    return avg_final_eval_reward