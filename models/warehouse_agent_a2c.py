from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from gymnasium import Env
from gymnasium.spaces import Discrete
from torch import Tensor
from torch.distributions import Categorical
from torch.optim import Adam
from tqdm import tqdm

from logger.logger import AppLogger
from models.actor_network import ActorNetwork
from models.critic_network import CriticNetwork
from utils.model_plotting import ModelPlotting
from warehouse_env.warehouse_env_3_A2C import WareHouseEnv3A2C


class WareHouseAgentA2C:
    def __init__(self) -> None:
        self._gamma: float = 0.95
        self._learning_rate: float = 3e-4
        self._entropy_coefficient: float = 0.08 #higher = more exploring 
        self._critic_coefficient: float = 0.5
        self._timestamp_string: str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

        # This is the number of extra iterations to run after loading the
        # resume checkpoint, not the model's total lifetime training count.
        self._total_training_iterations: int = 7000
        self._time_steps_per_batch: int =  5000 #5_000
        self._max_time_steps_per_episode: int = 500 #100 #2_500
        self._resume_from_latest_checkpoint: bool = True
        # Pin resume to the last pre-mask checkpoint so continuation training
        # does not build on the bad masked-policy run.
        self._resume_checkpoint_filename: str | None = None # "a2c_checkpoint_step_3550_2026_04_08_05_48_35.pt"

        self._environment_obj: Env = WareHouseEnv3A2C(render_mode=None)
        self._logger = AppLogger.get_logger(self.__class__.__name__)

        # Keep same 3-action setup as PPO for fair comparison
        # 0 = LEFT, 1 = RIGHT, 2 = FORWARD
        self._action_dimensions: int = Discrete(4).n

        self._device = self._get_device()

        self._actor_network: ActorNetwork = ActorNetwork(
            output_dimensions=self._action_dimensions,
            device=self._device
        )
        self._critic_network: CriticNetwork = CriticNetwork(
            device=self._device
        )

        self._actor_network_optimizer: Adam = Adam(
            params=self._actor_network.parameters(),
            lr=self._learning_rate
        )
        self._critic_network_optimizer: Adam = Adam(
            params=self._critic_network.parameters(),
            lr=self._learning_rate
        )
        self._model_plotting: ModelPlotting = ModelPlotting()

        self._training_rewards: list[float] = []
        self._training_time_steps: list[int] = []
        self._training_episode_numbers: list[int] = []
        self._training_episode_rewards: list[float] = []
        self._update_numbers: list[int] = []
        self._actor_loss_history: list[float] = []
        self._critic_loss_history: list[float] = []

        self._invalid_pickup_attempt_log_count: int = 0
        self._missed_pickup_opportunity_log_count: int = 0

    def train_agent(self) -> None:
        start_iteration: int = 0

        if self._resume_from_latest_checkpoint:
            start_iteration = self._load_checkpoint_to_resume()

        progress_bar: tqdm = tqdm(
            total=self._total_training_iterations,
            desc="Training Warehouse A2C Agent"
        )

        for iteration_offset in range(self._total_training_iterations):
            current_training_iteration = start_iteration + iteration_offset + 1

            final_training_iteration: int = start_iteration + self._total_training_iterations
            is_save_point: bool = self._is_save_point(
                current_training_iteration=current_training_iteration,
                final_training_iteration=final_training_iteration
            )

            if is_save_point:
                self._save_checkpoint(current_training_iteration=current_training_iteration)
                self._plot_rewards_by_episode()
                self._plot_rewards_by_time_step()
                self._plot_actor_and_critic_losses_by_update()

            (
                batch_observation_tensor,
                batch_actions_tensor,
                batch_returns_tensor,
                batch_episode_rewards_list,
                batch_episode_lengths_list,
                batch_episode_successes_list,
            ) = self._rollout()
            

            state_values_tensor: Tensor = self._critic_network(batch_observation_tensor).squeeze(-1)

            advantage_tensor: Tensor = batch_returns_tensor - state_values_tensor.detach()
            advantage_tensor = self._normalize_tensor(advantage_tensor)

            action_probabilities_tensor: Tensor = self._actor_network(batch_observation_tensor)
            categorical_distribution: Categorical = Categorical(probs=action_probabilities_tensor)

            log_probabilities_tensor: Tensor = categorical_distribution.log_prob(batch_actions_tensor)
            entropy_tensor: Tensor = categorical_distribution.entropy().mean()

            actor_loss_tensor: Tensor = -(log_probabilities_tensor * advantage_tensor).mean()
            # Entropy regularization keeps some exploration pressure on the
            # policy instead of collapsing too early to one repeated action.
            actor_loss_tensor = actor_loss_tensor - self._entropy_coefficient * entropy_tensor

            critic_loss_tensor: Tensor = torch.nn.functional.mse_loss(
                state_values_tensor,
                batch_returns_tensor
            )

            total_loss_tensor: Tensor = actor_loss_tensor + self._critic_coefficient * critic_loss_tensor

            self._actor_network_optimizer.zero_grad()
            self._critic_network_optimizer.zero_grad()

            total_loss_tensor.backward()

            self._actor_network_optimizer.step()
            self._critic_network_optimizer.step()

            mean_episode_reward: float = sum(batch_episode_rewards_list) / len(batch_episode_rewards_list)
            mean_episode_length: float = sum(batch_episode_lengths_list) / len(batch_episode_lengths_list)
            success_rate: float = sum(batch_episode_successes_list) / len(batch_episode_successes_list)

            self._update_numbers.append(current_training_iteration)
            self._actor_loss_history.append(float(actor_loss_tensor.item()))
            self._critic_loss_history.append(float(critic_loss_tensor.item()))

            self._logger.info(
                f"[A2C] Iteration={current_training_iteration} | "
                f"Actor Loss={actor_loss_tensor.item():.6f} | "
                f"Critic Loss={critic_loss_tensor.item():.6f} | "
                f"Entropy={self._entropy_coefficient:.6f} | "
                # f"Mean Discounted Return={mean_return:.6f} | "
                # f"Mean Value={mean_value:.6f} | "
                # f"Mean Episode Reward={mean_episode_reward:.4f} | "
                # f"Mean Episode Length={mean_episode_length:.2f} | "
                # f"Success Rate={success_rate:.2f}"
            )
            self._logger.info("=" * 100)

            progress_bar.update(1)

        # Always save the true final resumed iteration explicitly.
        final_training_iteration = start_iteration + self._total_training_iterations
        self._save_checkpoint(current_training_iteration=final_training_iteration)
        self._plot_rewards_by_episode()
        self._plot_rewards_by_time_step()
        self._plot_actor_and_critic_losses_by_update()
        progress_bar.close()
        self._environment_obj.close()

    def _rollout(self) -> tuple[Tensor, Tensor, Tensor, list[float], list[int], list[int]]:
        current_time_step: int = 0

        batch_observation_list: list[dict] = []
        batch_actions_list: list[int] = []
        batch_rewards_list: list[list[float]] = []
        global_time_step: int = len(self._training_time_steps)
        global_episode_number: int = len(self._training_episode_numbers)

        # ADD THESE
        batch_episode_rewards_list: list[float] = []
        batch_episode_lengths_list: list[int] = []
        batch_episode_successes_list: list[int] = []

        while current_time_step < self._time_steps_per_batch:
            episode_rewards_list: list[float] = []

            # ADD THESE
            episode_total_reward: float = 0.0
            episode_length: int = 0
            episode_success: int = 0

            observation_dict, info_dict = self._environment_obj.reset(seed=42 + current_time_step)

            for _ in range(self._max_time_steps_per_episode):
                if current_time_step >= self._time_steps_per_batch:
                    break

                batch_observation_list.append(observation_dict)

                action_int: int = self._get_action(observation_dict=observation_dict)

                observation_dict, reward, is_terminated, is_truncated, info_dict = self._environment_obj.step(
                    action_int
                )

                if info_dict.get("valid_pickup_location", False) and info_dict.get("missed_pickup_opportunity", False):
                    self._missed_pickup_opportunity_log_count += 1
                    if (
                        self._missed_pickup_opportunity_log_count <= 5
                        or self._missed_pickup_opportunity_log_count % 250 == 0
                    ):
                        self._logger.info(
                            f"[TRAIN MISSED PICKUP OPPORTUNITY #{self._missed_pickup_opportunity_log_count}] "
                            f"action={action_int} | reward={reward:.2f} | "
                            f"pos={self._environment_obj.agent_pos} | dir={self._environment_obj.agent_dir} | "
                            f"missed_pickup={info_dict.get('missed_pickup_opportunity', False)}"
                        )

                if info_dict.get("invalid_pickup_attempt", False):
                    self._invalid_pickup_attempt_log_count += 1
                    if (
                        self._invalid_pickup_attempt_log_count <= 3
                        or self._invalid_pickup_attempt_log_count % 1000 == 0
                    ):
                        self._logger.info(
                            f"[TRAIN INVALID PICKUP ATTEMPT #{self._invalid_pickup_attempt_log_count}] "
                            f"action={action_int} | reward={reward:.2f} | "
                            f"pos={self._environment_obj.agent_pos} | dir={self._environment_obj.agent_dir}"
                        )

                batch_actions_list.append(action_int)
                episode_rewards_list.append(float(reward))
                global_time_step += 1
                self._training_time_steps.append(global_time_step)
                self._training_rewards.append(float(reward))

                # ADD THESE
                episode_total_reward += float(reward)
                episode_length += 1

                current_time_step += 1
                if is_terminated:
                    episode_success = 1
               
                if is_terminated or is_truncated:
                    break

            batch_rewards_list.append(episode_rewards_list)

            # ADD THESE
            batch_episode_rewards_list.append(episode_total_reward)
            batch_episode_lengths_list.append(episode_length)
            batch_episode_successes_list.append(episode_success)
            global_episode_number += 1
            self._training_episode_numbers.append(global_episode_number)
            self._training_episode_rewards.append(episode_total_reward)

        batch_observation_tensor: Tensor = self._get_observations_tensor(
            batch_observation_list=batch_observation_list
        )

        batch_actions_tensor: Tensor = torch.tensor(
            data=batch_actions_list,
            dtype=torch.long,
            device=self._device
        )

        batch_returns_tensor: Tensor = self._get_returns_tensor(
            batch_rewards_list=batch_rewards_list
        )

        return (
            batch_observation_tensor,
            batch_actions_tensor,
            batch_returns_tensor,
            batch_episode_rewards_list,
            batch_episode_lengths_list,
            batch_episode_successes_list,
        )
       

   

    def _get_action(self, observation_dict: dict) -> int:
        observation_image_array: np.ndarray = observation_dict.get("image")

        observation_tensor: Tensor = torch.tensor(
            data=observation_image_array,
            dtype=torch.float32,
            device=self._device
        )

        action_probabilities_tensor: Tensor = self._actor_network(observation_tensor)

        categorical_distribution: Categorical = Categorical(probs=action_probabilities_tensor)

        action_tensor: Tensor = categorical_distribution.sample()

        return int(action_tensor.item())

    def _get_observations_tensor(self, batch_observation_list: list[dict]) -> Tensor:
        batch_observation_image_list: list[np.ndarray] = []

        for observation_dict in batch_observation_list:
            observation_image_array: np.ndarray = observation_dict.get("image")
            batch_observation_image_list.append(observation_image_array)

        batch_observation_image_array: np.ndarray = np.array(batch_observation_image_list)

        batch_observation_tensor: Tensor = torch.tensor(
            data=batch_observation_image_array,
            dtype=torch.float32,
            device=self._device
        )

        # (Batch, Height, Width, Channel) -> (Batch, Channel, Height, Width)
        batch_observation_tensor = batch_observation_tensor.permute(0, 3, 1, 2)

        return batch_observation_tensor

    def _get_returns_tensor(self, batch_rewards_list: list[list[float]]) -> Tensor:
        batch_discounted_returns_list: list[float] = []

        for episode_rewards_list in batch_rewards_list:
            discounted_return: float = 0.0
            episode_discounted_returns_reversed_list: list[float] = []

            for reward_value in reversed(episode_rewards_list):
                discounted_return = reward_value + self._gamma * discounted_return
                episode_discounted_returns_reversed_list.append(discounted_return)

            episode_discounted_returns_list: list[float] = list(
                reversed(episode_discounted_returns_reversed_list)
            )

            batch_discounted_returns_list.extend(episode_discounted_returns_list)

        batch_returns_tensor: Tensor = torch.tensor(
            data=batch_discounted_returns_list,
            dtype=torch.float32,
            device=self._device
        )

        return batch_returns_tensor

    @staticmethod
    def _normalize_tensor(input_tensor: Tensor) -> Tensor:
        return (input_tensor - input_tensor.mean()) / (input_tensor.std() + 1e-10)

    def _save_checkpoint(self, current_training_iteration: int) -> None:

        file_path: Path = self._get_file_path(current_training_iteration=current_training_iteration)

        checkpoint_dict: dict[str, int | OrderedDict | dict] = {
            "current_training_iteration": current_training_iteration,
            "actor_state_dict": self._actor_network.state_dict(),
            "critic_state_dict": self._critic_network.state_dict(),
            "actor_optimizer_state_dict": self._actor_network_optimizer.state_dict(),
            "critic_optimizer_state_dict": self._critic_network_optimizer.state_dict(),
        }

        torch.save(checkpoint_dict, file_path)

        self._logger.info("\n")
        self._logger.info("=" * 100)
        self._logger.info(f"Successfully saved: {file_path}")
        self._logger.info("=" * 100)

    def _plot_rewards_by_episode(self) -> None:
        actual_episode_number: int = self._training_episode_numbers[-1] if self._training_episode_numbers else 0
        file_path_reward_by_episode: Path = self._get_plot_file_path(
            file_path_str="rewards_by_episode",
            current_training_iteration=actual_episode_number,
            file_type_str="png"
        )
        self._model_plotting.plot_rewards_by_episode(
            file_path=file_path_reward_by_episode,
            training_episode_numbers=self._training_episode_numbers,
            training_episode_rewards=self._training_episode_rewards
        )

    def _plot_rewards_by_time_step(self) -> None:
        actual_time_step: int = self._training_time_steps[-1] if self._training_time_steps else 0
        file_path_rewards_by_timestep: Path = self._get_plot_file_path(
            file_path_str="rewards_by_time_step",
            current_training_iteration=actual_time_step,
            file_type_str="png"
        )
        self._model_plotting.plot_rewards_by_time_step(
            file_path=file_path_rewards_by_timestep,
            training_time_steps=self._training_time_steps,
            training_rewards=self._training_rewards
        )

    def _plot_actor_and_critic_losses_by_update(self) -> None:
        actual_update_number: int = self._update_numbers[-1] if self._update_numbers else 0
        file_path_actor_critic_loss: Path = self._get_plot_file_path(
            file_path_str="actor_critic_losses_by_update",
            current_training_iteration=actual_update_number,
            file_type_str="png"
        )
        self._model_plotting.plot_actor_and_critic_losses_by_update(
            file_path=file_path_actor_critic_loss,
            update_numbers=self._update_numbers,
            actor_loss_history=self._actor_loss_history,
            critic_loss_history=self._critic_loss_history
        )

    def _load_checkpoint_to_resume(self) -> int:
        checkpoint_directory_path: Path = Path("model_weights_a2c")

        if not checkpoint_directory_path.is_dir():
            return 0

        checkpoint_paths_list: list[Path] = sorted(
            checkpoint_directory_path.glob("*.pt"),
            key=lambda path: path.stat().st_mtime
        )

        if len(checkpoint_paths_list) == 0:
            return 0

        checkpoint_path_to_resume: Path

        if self._resume_checkpoint_filename is not None:
            candidate_checkpoint_path = checkpoint_directory_path / self._resume_checkpoint_filename
            if candidate_checkpoint_path.is_file():
                checkpoint_path_to_resume = candidate_checkpoint_path
            else:
                self._logger.info(
                    f"Requested resume checkpoint not found: {candidate_checkpoint_path}. "
                    f"Falling back to latest checkpoint."
                )
                checkpoint_path_to_resume = checkpoint_paths_list[-1]
        else:
            checkpoint_path_to_resume = checkpoint_paths_list[-1]

        checkpoint_dict: dict[str, int | OrderedDict | dict] = torch.load(
            checkpoint_path_to_resume,
            map_location=self._device
        )

        self._actor_network.load_state_dict(checkpoint_dict["actor_state_dict"])
        self._critic_network.load_state_dict(checkpoint_dict["critic_state_dict"])
        self._actor_network_optimizer.load_state_dict(checkpoint_dict["actor_optimizer_state_dict"])
        self._critic_network_optimizer.load_state_dict(checkpoint_dict["critic_optimizer_state_dict"])

        self._actor_network.to(self._device)
        self._critic_network.to(self._device)

        resumed_iteration: int = int(checkpoint_dict["current_training_iteration"])

        self._logger.info("=" * 100)
        self._logger.info(f"Resuming training from: {checkpoint_path_to_resume}")
        self._logger.info(f"Starting after iteration: {resumed_iteration}")
        self._logger.info("=" * 100)

        return resumed_iteration

    def _is_save_point(self, current_training_iteration: int, final_training_iteration: int) -> bool:
        is_checkpoint_save_point: bool = current_training_iteration % 500 == 0
        is_checkpoint_save_point_not_first_step: bool = current_training_iteration > 0
        is_end_of_training_save_point: bool = current_training_iteration == final_training_iteration

        return (is_checkpoint_save_point_not_first_step and is_checkpoint_save_point) or is_end_of_training_save_point

    def _get_file_path(self, current_training_iteration: int) -> Path:
        model_weights_directory_path: Path = Path("model_weights_a2c")
        model_weights_directory_path.mkdir(parents=True, exist_ok=True)
        
        self._logger.info("=" * 100)
        self._logger.info(f"Successfully created directory: {model_weights_directory_path}")
        self._logger.info("=" * 100)

        timestamp_string: str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        filename: str = f"a2c_checkpoint_step_{current_training_iteration}_{timestamp_string}.pt"

        return model_weights_directory_path / filename

    def _get_plot_file_path(self, file_path_str: str, current_training_iteration: int, file_type_str: str) -> Path:
        plotting_directory_path: Path = Path("model_plots_a2c")
        plotting_directory_path.mkdir(parents=True, exist_ok=True)

        filename_str: str = (
            f"{file_path_str}_{current_training_iteration}_{self._timestamp_string}.{file_type_str}"
        )
        return plotting_directory_path / filename_str

    def _get_device(self):
        if torch.cuda.is_available():
            self._logger.info("Using Device CUDA")
            return torch.device("cuda")
        if torch.mps.is_available():
            self._logger.info("Using Device MPS")
            return torch.device("mps")

        self._logger.info("Using Device CPU")
        return torch.device("cpu")
