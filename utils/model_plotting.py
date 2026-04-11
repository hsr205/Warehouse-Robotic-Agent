from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from logger.logger import AppLogger


class ModelPlotting:

    def __init__(self) -> None:
        self._logger = AppLogger.get_logger(self.__class__.__name__)

    def plot_rewards_by_time_step(self, file_path: Path, training_time_steps: list[int],
                                  training_rewards: list[float]) -> None:

        plt.figure(figsize=(12, 6))
        axis = plt.gca()

        axis.xaxis.set_major_formatter(FuncFormatter(self._format_large_numbers))
        axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

        if not training_time_steps or not training_rewards:
            self._logger.warning("No training history available to plot.")
            return

        x_values: np.ndarray = np.array(training_time_steps, dtype=np.int64)
        reward_values: np.ndarray = np.array(training_rewards, dtype=np.float32)

        # NOTE: Smooth the rewards so the chart displays the learning trend
        moving_average_window_size: int = 1_000

        if len(reward_values) >= moving_average_window_size:
            kernel: np.ndarray = np.ones(moving_average_window_size, dtype=np.float32) / moving_average_window_size
            smoothed_rewards: np.ndarray = np.convolve(reward_values, kernel, mode="valid")
            smoothed_x_values: np.ndarray = x_values[moving_average_window_size - 1:]

            plt.plot(smoothed_x_values, smoothed_rewards, label=f"Reward Moving Average ({moving_average_window_size})")
        else:
            # NOTE: Fallback if not enough data for smoothing
            plt.plot(x_values, reward_values, label="Raw Rewards")

        plt.ylabel("Reward")
        plt.xlabel("Number of Timesteps Taken")
        plt.title("Rewards by Total Training Timesteps")

        plt.tight_layout()

        self._logger.info("=" * 100)
        self._logger.info(f"Saving plot: {file_path}")
        plt.savefig(file_path)
        plt.close()
        self._logger.info(f"Successfully saved plot: {file_path}")
        self._logger.info("=" * 100)

    def plot_rewards_by_episode(self, file_path: Path, training_episode_numbers: list[int],
                                training_episode_rewards: list[float]) -> None:

        if not training_episode_numbers or not training_episode_rewards:
            self._logger.warning("No episode reward history available to plot.")
            return

        plt.figure(figsize=(12, 6))
        axis = plt.gca()

        episode_x_values: np.ndarray = np.array(training_episode_numbers, dtype=np.int64)
        episode_reward_values: np.ndarray = np.array(training_episode_rewards, dtype=np.float32)

        # CHANGED: Smooth episode returns so the trend is easier to interpret
        moving_average_window_size: int = 100

        if len(episode_reward_values) >= moving_average_window_size:
            kernel: np.ndarray = np.ones(moving_average_window_size, dtype=np.float32) / moving_average_window_size
            smoothed_episode_rewards: np.ndarray = np.convolve(episode_reward_values, kernel, mode="valid")
            smoothed_episode_x_values: np.ndarray = episode_x_values[moving_average_window_size - 1:]

            axis.plot(
                smoothed_episode_x_values,
                smoothed_episode_rewards,
                label=f"Episode Reward Moving Average ({moving_average_window_size})"
            )
        else:
            axis.plot(
                episode_x_values,
                episode_reward_values,
                label="Episode Rewards"
            )

        axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
        axis.set_xlabel("Episode Number")
        axis.set_ylabel("Total Reward Per Episode")
        axis.set_title("Rewards by Episode")
        axis.legend()

        plt.tight_layout()

        self._logger.info("=" * 100)
        self._logger.info(f"Saving plot: {file_path}")
        plt.savefig(file_path)
        plt.close()
        self._logger.info(f"Successfully saved plot: {file_path}")
        self._logger.info("=" * 100)

    def plot_actor_and_critic_losses_by_update(self, file_path: Path,
                                               update_numbers: list[int],
                                               actor_loss_history: list[float],
                                               critic_loss_history: list[float]) -> None:

        if not update_numbers or not actor_loss_history or not critic_loss_history:
            self._logger.warning("No actor/critic loss history available to plot.")
            return

        plt.figure(figsize=(12, 6))
        axis = plt.gca()

        update_x_values: np.ndarray = np.array(update_numbers, dtype=np.int64)
        actor_loss_values: np.ndarray = np.array(actor_loss_history, dtype=np.float32)
        critic_loss_values: np.ndarray = np.array(critic_loss_history, dtype=np.float32)

        axis.plot(
            update_x_values,
            actor_loss_values,
            label="Actor Loss"
        )

        axis.plot(
            update_x_values,
            critic_loss_values,
            label="Critic Loss"
        )

        axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
        axis.set_xlabel("Update Number")
        axis.set_ylabel("Loss")
        axis.set_title("Actor and Critic Losses by Update")
        axis.legend()

        plt.tight_layout()

        self._logger.info("=" * 100)
        self._logger.info(f"Saving plot: {file_path}")
        plt.savefig(file_path)
        plt.close()
        self._logger.info(f"Successfully saved plot: {file_path}")
        self._logger.info("=" * 100)

    def _format_large_numbers(self, x, pos):
        if x >= 1_000_000:
            return f"{x / 1_000_000:.1f}M"
        if x >= 1_000:
            return f"{int(x / 1_000)}k"
        return str(int(x))
