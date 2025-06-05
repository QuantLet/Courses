# src/utils/plotting.py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import os
import logging
from typing import Dict, Any, List, Optional  # For type hinting
from omegaconf import DictConfig  # For cfg type hint

# Optional dependency for Optuna visualizations
import optuna  # Ensure optuna is installed if you want to use its plotting features

# For Optuna plots (optional dependency)
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_slice,
)
import plotly  # Ensure plotly is importable if using optuna plots

# Import PLOTLY_OK from common_utils to check if plotly is available
from .common_utils import (
    PLOTLY_OK,
)  # Assuming common_utils is in the same 'utils' package

log = logging.getLogger(__name__)

# Consistent plot styling
plt.style.use("seaborn-v0_8-whitegrid")
# You can set global rcParams here if you want consistent font sizes etc. for all plots
# plt.rcParams.update({'font.size': 12, 'figure.dpi': 100})


def plot_training_results(
    results_dict: Dict[str, List[Any]],
    cfg_training_params: DictConfig,  # e.g., cfg.training or relevant parts of main cfg
    save_dir: str,
    policy_name_prefix: str = "Agent",
) -> None:
    """
    Plots training metrics: rewards, episode lengths, losses, and epsilon decay.

    Args:
        results_dict: Dictionary containing lists for 'rewards', 'lengths',
                      'losses' (or 'actor_losses', 'critic_losses'), 'epsilon'.
        cfg_training_params: Config object containing training parameters like 'num_episodes'.
        save_dir: Directory to save the plots.
        policy_name_prefix: Prefix for plot titles and filenames (e.g., "DQN", "A2C").
    """
    log.info(f"Generating training plots for {policy_name_prefix} in {save_dir}")
    os.makedirs(save_dir, exist_ok=True)

    num_total_episodes = cfg_training_params.get(
        "num_episodes", len(results_dict.get("rewards", []))
    )
    # Smoother window size, ensuring it's at least 1 and not too large for few episodes
    window_size = max(
        1, min(50, num_total_episodes // 20 if num_total_episodes > 20 else 5)
    )

    # 1. Plot Cumulative Rewards
    if "rewards" in results_dict and results_dict["rewards"]:
        try:
            plt.figure(figsize=(12, 6))
            rewards = results_dict["rewards"]
            episodes_axis = range(1, len(rewards) + 1)
            plt.plot(
                episodes_axis,
                rewards,
                label=f"{policy_name_prefix} Episode Reward",
                alpha=0.4,
                lw=1.5,
                color="C0",
            )
            if len(rewards) >= window_size:
                rolling_rewards = (
                    pd.Series(rewards).rolling(window=window_size, min_periods=1).mean()
                )
                plt.plot(
                    episodes_axis,
                    rolling_rewards,
                    label=f"Rolling Avg (w={window_size})",
                    color="C0",
                    lw=2.5,
                )

            plt.xlabel("Episode")
            plt.ylabel("Cumulative Reward")
            plt.title(
                f"{policy_name_prefix} Training Rewards (Episodes: {num_total_episodes})"
            )
            plt.legend(loc="best")
            plt.grid(True, which="both", linestyle="--", linewidth=0.5)
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    save_dir, f"{policy_name_prefix.lower()}_training_rewards.png"
                ),
                dpi=150,
            )
            plt.close()
            log.debug(f"Saved {policy_name_prefix} training rewards plot.")
        except Exception as e:
            log.error(
                f"Error generating {policy_name_prefix} rewards plot: {e}",
                exc_info=True,
            )
    else:
        log.info(f"No 'rewards' data to plot for {policy_name_prefix}.")

    # 2. Plot Episode Lengths
    if "lengths" in results_dict and results_dict["lengths"]:
        try:
            plt.figure(figsize=(12, 6))
            lengths = results_dict["lengths"]
            episodes_axis = range(1, len(lengths) + 1)
            plt.plot(
                episodes_axis,
                lengths,
                label=f"{policy_name_prefix} Episode Length",
                alpha=0.4,
                lw=1.5,
                color="C1",
            )
            if len(lengths) >= window_size:
                rolling_lengths = (
                    pd.Series(lengths).rolling(window=window_size, min_periods=1).mean()
                )
                plt.plot(
                    episodes_axis,
                    rolling_lengths,
                    label=f"Rolling Avg (w={window_size})",
                    color="C1",
                    lw=2.5,
                )

            plt.xlabel("Episode")
            plt.ylabel("Number of Steps")
            plt.title(f"{policy_name_prefix} Training Episode Lengths")
            plt.legend(loc="best")
            plt.grid(True, which="both", linestyle="--", linewidth=0.5)
            plt.ylim(bottom=0)  # Ensure y-axis starts at 0 for lengths
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    save_dir, f"{policy_name_prefix.lower()}_training_lengths.png"
                ),
                dpi=150,
            )
            plt.close()
            log.debug(f"Saved {policy_name_prefix} training lengths plot.")
        except Exception as e:
            log.error(
                f"Error generating {policy_name_prefix} lengths plot: {e}",
                exc_info=True,
            )
    else:
        log.info(f"No 'lengths' data to plot for {policy_name_prefix}.")

    # 3. Plot Training Loss (Handles different loss keys)
    loss_keys_found = []
    if (
        "losses" in results_dict and results_dict["losses"]
    ):  # For DQN/DDQN (single loss)
        loss_keys_found.append(("losses", "Overall Loss", "C2"))
    if "actor_losses" in results_dict and results_dict["actor_losses"]:  # For A2C
        loss_keys_found.append(("actor_losses", "Actor Loss", "C3"))
    if "critic_losses" in results_dict and results_dict["critic_losses"]:  # For A2C
        loss_keys_found.append(("critic_losses", "Critic Loss", "C4"))

    if loss_keys_found:
        try:
            plt.figure(figsize=(12, 6))
            for key, label, color_code in loss_keys_found:
                losses = [
                    l for l in results_dict[key] if l is not None and not np.isnan(l)
                ]  # Filter out NaNs for plotting
                if not losses:
                    log.info(f"No valid '{key}' data to plot for {policy_name_prefix}.")
                    continue

                # X-axis should correspond to episodes where learning actually occurred.
                # If losses are recorded per learning step, and learning happens less than once per episode,
                # the x-axis needs careful handling. Assuming losses are per episode averages for now.
                episodes_axis_loss = range(1, len(losses) + 1)

                plt.plot(
                    episodes_axis_loss,
                    losses,
                    label=f"{policy_name_prefix} Avg {label}",
                    alpha=0.4,
                    lw=1.5,
                    color=color_code,
                )
                if len(losses) >= window_size:
                    rolling_loss = (
                        pd.Series(losses)
                        .rolling(window=window_size, min_periods=1)
                        .mean()
                    )
                    plt.plot(
                        episodes_axis_loss,
                        rolling_loss,
                        label=f"Rolling Avg {label} (w={window_size})",
                        color=color_code,
                        lw=2.5,
                    )

            plt.xlabel("Episode (where learning occurred)")
            plt.ylabel("Average Loss Value")
            plt.title(f"{policy_name_prefix} Training Losses")
            plt.legend(loc="best")
            plt.grid(True, which="both", linestyle="--", linewidth=0.5)
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    save_dir, f"{policy_name_prefix.lower()}_training_loss.png"
                ),
                dpi=150,
            )
            plt.close()
            log.debug(f"Saved {policy_name_prefix} training loss plot(s).")
        except Exception as e:
            log.error(
                f"Error generating {policy_name_prefix} loss plot: {e}", exc_info=True
            )
    else:
        log.info(
            f"No 'loss' data (losses, actor_losses, critic_losses) to plot for {policy_name_prefix}."
        )

    # 4. Plot Epsilon Decay (if applicable)
    if "epsilon" in results_dict and results_dict["epsilon"]:
        try:
            plt.figure(figsize=(10, 5))
            epsilon_values = results_dict["epsilon"]
            episodes_axis = range(1, len(epsilon_values) + 1)
            plt.plot(
                episodes_axis, epsilon_values, label="Epsilon Value", color="C5", lw=2
            )
            plt.xlabel("Episode")
            plt.ylabel("Epsilon")
            plt.title(f"{policy_name_prefix} Epsilon Decay Schedule")
            plt.grid(True, which="both", linestyle="--", linewidth=0.5)
            plt.ylim(0, 1.05)  # Epsilon is typically between 0 and 1
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    save_dir, f"{policy_name_prefix.lower()}_epsilon_decay.png"
                ),
                dpi=150,
            )
            plt.close()
            log.debug(f"Saved {policy_name_prefix} epsilon decay plot.")
        except Exception as e:
            log.error(
                f"Error generating {policy_name_prefix} epsilon plot: {e}",
                exc_info=True,
            )
    else:
        log.info(f"No 'epsilon' data to plot for {policy_name_prefix}.")


def plot_evaluation_comparison(
    all_eval_results: Dict[str, Dict[str, Any]],
    num_eval_episodes_for_title: int,
    save_dir: str,
) -> None:
    """
    Plots a bar chart comparing average evaluation rewards of different policies.

    Args:
        all_eval_results: Dictionary where keys are policy names and values are
                          dicts containing 'avg_reward', 'std_reward', 'rewards' list.
        num_eval_episodes_for_title: Number of evaluation episodes, for the plot title.
        save_dir: Directory to save the plot.
    """
    log.info(f"Generating policy evaluation comparison plot in {save_dir}")
    os.makedirs(save_dir, exist_ok=True)

    # Filter out policies with no valid results (e.g., MVO if CVXPY fails and returns None/NaNs)
    valid_policies_data = []
    for name, results in all_eval_results.items():
        if (
            results
            and "rewards" in results
            and results["rewards"]
            and not np.all(np.isnan(results["rewards"]))
        ):
            avg_r = results.get("avg_reward", np.mean(results["rewards"]))
            std_r = results.get("std_reward", np.std(results["rewards"]))
            if not (np.isnan(avg_r) or np.isnan(std_r)):
                valid_policies_data.append(
                    {"name": name, "avg_reward": avg_r, "std_reward": std_r}
                )

    if not valid_policies_data:
        log.warning("No valid evaluation results to plot for comparison.")
        return

    df_plot = pd.DataFrame(valid_policies_data).sort_values(
        by="avg_reward", ascending=False
    )

    try:
        plt.figure(
            figsize=(max(8, len(df_plot) * 1.5), 7)
        )  # Adjust width based on num policies

        colors = []
        palette = sns.color_palette("viridis", n_colors=len(df_plot))
        for i, policy_name in enumerate(df_plot["name"]):
            if "DQN" in policy_name or "A2C" in policy_name or "DDQN" in policy_name:
                colors.append("darkgreen")
            elif policy_name == "Random":
                colors.append("grey")
            else:  # Heuristics
                colors.append(palette[i % len(palette)])

        bars = plt.bar(
            df_plot["name"],
            df_plot["avg_reward"],
            yerr=df_plot["std_reward"],
            capsize=5,
            color=colors,
            alpha=0.85,
        )

        plt.ylabel("Average Cumulative Reward")
        plt.title(
            f"Policy Performance Comparison ({num_eval_episodes_for_title} Evaluation Episodes)"
        )
        plt.xticks(rotation=30, ha="right", fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid(axis="y", linestyle="--", alpha=0.7)

        # Add text labels on bars
        for bar in bars:
            yval = bar.get_height()
            y_err = (
                0  # Find corresponding error bar if needed for precise text placement
            )
            # This requires matching bar to df_plot entry, simplified for now
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                yval
                + np.sign(yval) * 0.02 * np.abs(yval)
                + np.sign(yval) * y_err * 0.1,  # Adjust offset
                f"{yval:.2f}",
                ha="center",
                va="bottom" if yval >= 0 else "top",
                fontsize=9,
            )

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "policy_evaluation_comparison.png"), dpi=150)
        plt.close()
        log.debug("Saved policy evaluation comparison plot.")
    except Exception as e:
        log.error(f"Error generating evaluation comparison plot: {e}", exc_info=True)


def plot_optuna_visualizations(
    study: "optuna.Study", save_dir: str
) -> None:  # Use string literal for optuna.Study
    """Plots Optuna study results if Plotly is available."""
    if not PLOTLY_OK:
        log.info("Plotly not installed. Skipping Optuna visualizations.")
        return
    # Conditional import if PLOTLY_OK is True
    from optuna.visualization import (
        plot_optimization_history,
        plot_param_importances,
        plot_slice,
    )

    log.info(f"Generating Optuna plots for study '{study.study_name}' in {save_dir}")
    os.makedirs(save_dir, exist_ok=True)

    try:
        # 1. Optimization History
        fig_history = plot_optimization_history(study)
        fig_history.write_image(
            os.path.join(save_dir, "optuna_optimization_history.png"), scale=2
        )
        log.debug("Saved Optuna optimization history plot.")

        # 2. Parameter Importance (only if study has completed trials and parameters)
        if (
            study.trials
            and any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
            and study.best_params
        ):
            try:
                fig_importance = plot_param_importances(study)
                fig_importance.write_image(
                    os.path.join(save_dir, "optuna_param_importances.png"), scale=2
                )
                log.debug("Saved Optuna parameter importances plot.")
            except ValueError as ve_imp:  # Can happen if only one param or no completed trials with params
                log.warning(
                    f"Could not generate Optuna parameter importance plot: {ve_imp}"
                )

            # 3. Slice Plot (requires 2+ hyperparameters)
            if len(study.best_params) >= 2:
                try:
                    fig_slice = plot_slice(study)
                    fig_slice.write_image(
                        os.path.join(save_dir, "optuna_slice_plot.png"), scale=2
                    )
                    log.debug("Saved Optuna slice plot.")
                except Exception as slice_e:  # Catches various errors from plot_slice
                    log.warning(f"Could not generate Optuna slice plot: {slice_e}")
            else:
                log.info(
                    "Skipping Optuna slice plot: Not enough hyperparameters tuned."
                )
        else:
            log.info(
                "Skipping Optuna parameter importance and slice plots: No completed trials or parameters."
            )

    except ImportError:  # Should be caught by PLOTLY_OK but as a safeguard
        log.error(
            "Plotly or Optuna visualization components not found, though PLOTLY_OK was True. Skipping Optuna plots."
        )
    except Exception as e:
        log.error(f"Error generating Optuna plots: {e}", exc_info=True)
