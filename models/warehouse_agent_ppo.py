from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import torch
from gymnasium.spaces import Discrete
from torch import nn, Tensor
from torch.distributions import Categorical
from torch.optim import Adam
from tqdm import tqdm

from logger.logger import AppLogger
from models.actor_network import ActorNetwork
from models.critic_network import CriticNetwork
from utils.model_plotting import ModelPlotting
from warehouse_env.warehouse_env import WareHouseEnv
from warehouse_env.warehouse_env_2 import WareHouseEnv2
from warehouse_env.warehouse_env_3 import WareHouseEnv3


class WareHouseAgentPPO:

    def __init__(self, environment_obj: WareHouseEnv | WareHouseEnv2 | WareHouseEnv3,
                 total_actions_taken_during_training_episode: int, batch_size_before_policy_update: int) -> None:
        # NOTE: CLIP acts as a threshold for making sure our policy
        #       does not change too dramatically when conducting SGA
        self._clip: float = 0.2
        self._gamma: float = 0.95
        self._environment_obj = environment_obj

        self._learning_rate: float = 3e-4
        self._entropy_coefficient: float = 0.075
        self._num_updates_per_iteration: int = 5
        self._max_time_steps_per_episode: int = 100
        self._logger = AppLogger.get_logger(self.__class__.__name__)
        self._timestamp_string: str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self._action_dimensions = self._environment_obj.action_space = Discrete(4).n

        self._total_actions_taken_during_training: int = total_actions_taken_during_training_episode
        self._time_steps_per_batch_before_policy_update: int = batch_size_before_policy_update

        self._device = self._get_device()
        self._actor_network: ActorNetwork = ActorNetwork(output_dimensions=self._action_dimensions,
                                                         device=self._device)
        self._critic_network: CriticNetwork = CriticNetwork(device=self._device)

        self._actor_network_optimizer: Adam = Adam(params=self._actor_network.parameters(),
                                                   lr=self._learning_rate)

        self._critic_network_optimizer: Adam = Adam(params=self._critic_network.parameters(),
                                                    lr=self._learning_rate)

        self._model_plotting: ModelPlotting = ModelPlotting()

        self._training_rewards: list[float] = []
        self._training_time_steps: list[int] = []

        self._training_episode_numbers: list[int] = []
        self._training_episode_rewards: list[float] = []

        self._update_numbers: list[int] = []
        self._actor_loss_history: list[float] = []
        self._critic_loss_history: list[float] = []

    def train_agent(self) -> None:

        current_training_iteration: int = 0
        start_time: datetime = datetime.now(ZoneInfo("America/New_York"))
        progress_bar: tqdm = tqdm(total=self._total_actions_taken_during_training, desc="Training Warehouse PPO Agent")

        while current_training_iteration <= self._total_actions_taken_during_training:

            is_save_point: bool = self._is_save_point(current_training_iteration=current_training_iteration)

            if is_save_point:
                self._save_checkpoint(current_training_iteration=current_training_iteration, start_time=start_time)

            batch_observation_tensor, batch_actions_tensor, batch_rewards_tensor, batch_length_tensor, batch_log_probability_tensor = self._rollout()

            # NOTE: Calculates V from the critic_network, and the log probabilities for to minimize the following formula:
            # FORMULA: π_theta(a_t,s_t) / π_theta_k(a_t,s_t):
            # Old Policy - π_theta(a,s)
            # New Policy - π_theta_k(a,s)
            v_tensor, _, _ = self._evaluate_agent(batch_observation_tensor=batch_observation_tensor,
                                                  batch_actions_tensor=batch_actions_tensor)

            # NOTE: Calculates the advantage tensor
            advantage_value_tensor: Tensor = self._get_normalized_advantage_value(
                batch_rewards_tensor=batch_rewards_tensor,
                v_tensor=v_tensor)

            actor_network_loss_tensor: Tensor | None = None
            critic_network_loss_tensor: Tensor | None = None

            for update_index_within_iteration in range(0, self._num_updates_per_iteration):
                v_tensor, current_log_probabilities_tensor, entropy_tensor = self._evaluate_agent(
                    batch_observation_tensor=batch_observation_tensor,
                    batch_actions_tensor=batch_actions_tensor

                )

                # NOTE: This ratio is simply: π_theta(a_t | s_t) / π_theta_k(a_t | s_t)
                ratios_tensor: Tensor = torch.exp(
                    input=(current_log_probabilities_tensor - batch_log_probability_tensor))

                # NOTE: This is: π_theta(a_t | s_t)
                surrogate_loss_tensor_1: Tensor = ratios_tensor * advantage_value_tensor

                # NOTE: The following prevents the ratio from take too large a step
                #       This is one if not thee core principle of PPO
                lower_bound_clip_value: float = 1 - self._clip
                upper_bound_clip_value: float = 1 + self._clip

                # NOTE: This is: π_theta_k(a_t | s_t)
                surrogate_loss_tensor_2: Tensor = torch.clamp(ratios_tensor, lower_bound_clip_value,
                                                              upper_bound_clip_value) * advantage_value_tensor
                # NOTE: Maximizes loss through negation
                #       this will be optimized by Adam later and will improve performance
                # NOTE: The entropy coefficient is leverage as an exploration bonus
                # NOTE: Un-Clipped -> r(θ)A
                # NOTE: Clipped -> min(clip(r(θ),0.8,1.2) * advantage)
                actor_network_loss_tensor: Tensor = -torch.min(surrogate_loss_tensor_1,
                                                               surrogate_loss_tensor_2).mean() - self._entropy_coefficient * entropy_tensor.mean()

                # NOTE: Zeroes out the gradient
                #       Conducts a backward propagation of the calculated loss
                #       Leverages the optimizer to take a step (determined by the learning rate) forward
                self._actor_network_optimizer.zero_grad()

                # NOTE: Retains the computation graph for later back propagation
                actor_network_loss_tensor.backward(retain_graph=True)
                self._actor_network_optimizer.step()

                critic_network_loss_tensor: Tensor = nn.MSELoss()(v_tensor, batch_rewards_tensor)

                self._critic_network_optimizer.zero_grad()
                critic_network_loss_tensor.backward()
                self._critic_network_optimizer.step()

                current_update_number: int = len(self._update_numbers) + 1
                self._update_numbers.append(current_update_number)
                self._actor_loss_history.append(actor_network_loss_tensor.item())
                self._critic_loss_history.append(critic_network_loss_tensor.item())

            progress_bar.update(1)

            self._update_progress_bar(progress_bar=progress_bar, actor_network_loss_tensor=actor_network_loss_tensor,
                                      critic_network_loss_tensor=critic_network_loss_tensor)

            current_training_iteration += 1

        progress_bar.close()

    def _update_progress_bar(self, progress_bar: tqdm, actor_network_loss_tensor: Tensor,
                             critic_network_loss_tensor: Tensor) -> None:
        if actor_network_loss_tensor is not None and critic_network_loss_tensor is not None:
            progress_bar.set_postfix({
                "actor_loss": actor_network_loss_tensor.item(),
                "critic_loss": critic_network_loss_tensor.item()
            })

    def _is_save_point(self, current_training_iteration: int) -> bool:
        is_checkpoint_save_point: bool = current_training_iteration % 500 == 0
        is_checkpoint_save_point_not_first_step: bool = current_training_iteration > 0
        is_end_of_training_save_point: bool = current_training_iteration == self._total_actions_taken_during_training

        is_save_point: bool = is_checkpoint_save_point_not_first_step and is_checkpoint_save_point or is_end_of_training_save_point

        return is_save_point

    def _evaluate_agent(self, batch_observation_tensor: Tensor, batch_actions_tensor: Tensor) -> tuple[
        Tensor, Tensor, Tensor]:

        # NOTE: Removes only the final dimension, not an entire flattening
        #       Estimates V(s) for each state
        v_tensor: Tensor = self._critic_network(batch_observation_tensor).squeeze(-1)

        action_probabilities_tensor: Tensor = self._actor_network(batch_observation_tensor)

        categorical_distribution: Categorical = Categorical(probs=action_probabilities_tensor)

        # NOTE: We sample from the actions available in categorical_distribution
        batch_actions_tensor = batch_actions_tensor.to(self._device).long()

        if batch_actions_tensor.dim() > 1:
            batch_actions_tensor = batch_actions_tensor.squeeze(-1)

        log_probabilities_tensor: Tensor = categorical_distribution.log_prob(batch_actions_tensor)

        entropy_tensor: Tensor = categorical_distribution.entropy()

        return v_tensor, log_probabilities_tensor, entropy_tensor

    def _rollout(self) -> tuple:

        current_time_step: int = 0
        batch_length_list: list[int] = []
        batch_actions_list: list[int] = []
        batch_observation_list: list[dict] = []
        batch_rewards_list: list[list[float]] = []
        batch_log_probability_list: list[float] = []
        global_time_step: int = len(self._training_time_steps)
        global_episode_number: int = len(self._training_episode_numbers)

        while current_time_step < self._time_steps_per_batch_before_policy_update:

            episode_length: int = 0
            episode_rewards: list[float] = []
            observation_dict, info_dict = self._environment_obj.reset(seed=42 + current_time_step)

            for _ in range(0, self._max_time_steps_per_episode):

                if current_time_step >= self._time_steps_per_batch_before_policy_update:
                    break

                batch_observation_list.append(observation_dict)

                action_int, log_probability = self._get_action(observation_dict=observation_dict)

                observation_dict, reward, is_terminated, is_truncated, info_dict = self._environment_obj.step(
                    action_int
                )

                is_done: bool = is_terminated or is_truncated

                global_time_step += 1

                self._training_time_steps.append(global_time_step)
                self._training_rewards.append(float(reward))

                episode_rewards.append(float(reward))
                batch_actions_list.append(action_int)
                batch_log_probability_list.append(log_probability)

                current_time_step += 1
                episode_length += 1

                if is_done:
                    break

            batch_length_list.append(episode_length + 1)
            batch_rewards_list.append(episode_rewards)

            episode_total_reward: float = sum(episode_rewards)

            global_episode_number += 1
            self._training_episode_numbers.append(global_episode_number)
            self._training_episode_rewards.append(episode_total_reward)

        batch_observation_tensor: Tensor = self._get_observations_tensor(batch_observation_list=batch_observation_list)

        batch_actions_tensor: Tensor = torch.tensor(data=batch_actions_list,
                                                    dtype=torch.int,
                                                    device=self._device)

        batch_rewards_tensor: Tensor = self._get_rewards_tensor(batch_rewards_list=batch_rewards_list)

        batch_length_tensor: Tensor = torch.tensor(data=batch_length_list,
                                                   dtype=torch.int,
                                                   device=self._device)

        batch_log_probability_tensor: Tensor = torch.tensor(data=batch_log_probability_list,
                                                            dtype=torch.float32,
                                                            device=self._device)

        self._environment_obj.close()

        return batch_observation_tensor, batch_actions_tensor, batch_rewards_tensor, batch_length_tensor, batch_log_probability_tensor

    def _get_normalized_advantage_value(self, batch_rewards_tensor: Tensor, v_tensor: Tensor) -> Tensor:

        advantage_value_tensor: Tensor = batch_rewards_tensor - v_tensor.detach()

        numerator_value = advantage_value_tensor - advantage_value_tensor.mean()
        denominator_value = advantage_value_tensor.std() + 1e-10

        return numerator_value / denominator_value

    def _get_observations_tensor(self, batch_observation_list: list[dict]) -> Tensor:

        batch_observation_image_list: list[np.ndarray] = []

        for observation_dict in batch_observation_list:
            observation_image_array: np.ndarray = observation_dict.get('image')
            batch_observation_image_list.append(observation_image_array)

        batch_observation_image_array: np.ndarray = np.array(batch_observation_image_list)

        batch_observation_tensor: Tensor = torch.tensor(data=batch_observation_image_array,
                                                        dtype=torch.float32,
                                                        device=self._device)

        # NOTE: Convert from (Batch, Height, Width, Channel) to (Batch, Channel, Height, Width)
        batch_observation_tensor = batch_observation_tensor.permute(0, 3, 1, 2)

        return batch_observation_tensor

    def _get_rewards_tensor(self, batch_rewards_list: list[list[float]]) -> Tensor:

        batch_discounted_rewards_list: list[float] = []

        for episode_rewards_list in batch_rewards_list:
            discounted_reward: float = 0.0
            episode_discounted_rewards_reversed_list: list[float] = []

            for reward_value in reversed(episode_rewards_list):
                discounted_reward = reward_value + discounted_reward * self._gamma
                episode_discounted_rewards_reversed_list.append(discounted_reward)

            episode_discounted_rewards_list: list[float] = list(reversed(episode_discounted_rewards_reversed_list))

            batch_discounted_rewards_list.extend(episode_discounted_rewards_list)

        batch_rewards_tensor = torch.tensor(data=batch_discounted_rewards_list, dtype=torch.float32,
                                            device=self._device)

        return batch_rewards_tensor

    def _get_action(self, observation_dict: dict) -> tuple[int, float]:

        observation_image_array: np.ndarray = observation_dict.get("image")

        observation_tensor: Tensor = torch.tensor(data=observation_image_array,
                                                  dtype=torch.float32,
                                                  device=self._device)

        # NOTE: Conducts a forward pass through the actor_network in order to retrieve the soft_max probability distribution
        action_probabilities: Tensor = self._actor_network(observation_tensor)

        # NOTE: Takes the action_probabilities and converts them back into categorical actions, values 0-6
        categorical_distribution: Categorical = Categorical(probs=action_probabilities)

        # NOTE: We sample from the actions available in categorical_distribution
        action_tensor: Tensor = categorical_distribution.sample()

        log_probabilities_tensor: Tensor = categorical_distribution.log_prob(action_tensor)

        return action_tensor.item(), log_probabilities_tensor.item()

    def _save_checkpoint(self, current_training_iteration: int, start_time: datetime) -> None:

        file_path: Path = self._get_file_path(file_path_str="checkpoint_step",
                                              current_training_iteration=current_training_iteration,
                                              file_type_str="pt")

        checkpoint_dict: dict[str, int | OrderedDict | dict] = {
            "clip": self._clip,
            "learning_rate": self._learning_rate,
            "entropy_coefficient": self._entropy_coefficient,
            "num_updates_per_iteration": self._num_updates_per_iteration,
            "max_time_steps_per_episode": self._max_time_steps_per_episode,
            "total_actions_taken_during_training": self._total_actions_taken_during_training,
            "time_steps_per_batch_before_policy_update": self._time_steps_per_batch_before_policy_update,
            "current_training_iteration": current_training_iteration,
            "actor_state_dict": self._actor_network.state_dict(),
            "critic_state_dict": self._critic_network.state_dict(),
            "actor_optimizer_state_dict": self._actor_network_optimizer.state_dict(),
            "critic_optimizer_state_dict": self._critic_network_optimizer.state_dict(),
        }

        torch.save(checkpoint_dict, file_path)

        self._plot_rewards_by_episode()
        self._plot_rewards_by_time_step()
        self._plot_actor_and_critic_losses_by_update()

        self._display_save_checkpoint_logger_statements(file_path=file_path, start_time=start_time)

    def _plot_rewards_by_episode(self) -> None:
        actual_episode_number: int = self._training_episode_numbers[-1] if self._training_episode_numbers else 0

        file_path_reward_by_episode: Path = self._get_file_path(
            file_path_str="rewards_by_episode",
            current_training_iteration=actual_episode_number,
            file_type_str="png"
        )

        self._model_plotting.plot_rewards_by_episode(file_path=file_path_reward_by_episode,
                                                     training_episode_numbers=self._training_episode_numbers,
                                                     training_episode_rewards=self._training_episode_rewards)

    def _plot_rewards_by_time_step(self) -> None:
        actual_time_step: int = self._training_time_steps[-1] if self._training_time_steps else 0

        file_path_rewards_by_timestep: Path = self._get_file_path(
            file_path_str="rewards_by_time_step",
            current_training_iteration=actual_time_step,
            file_type_str="png"
        )

        self._model_plotting.plot_rewards_by_time_step(file_path=file_path_rewards_by_timestep,
                                                       training_time_steps=self._training_time_steps,
                                                       training_rewards=self._training_rewards)

    def _plot_actor_and_critic_losses_by_update(self) -> None:

        actual_update_number: int = self._update_numbers[-1] if self._update_numbers else 0
        file_path_actor_critic_loss: Path = self._get_file_path(
            file_path_str="actor_critic_losses_by_update",
            current_training_iteration=actual_update_number,
            file_type_str="png"
        )

        self._model_plotting.plot_actor_and_critic_losses_by_update(file_path=file_path_actor_critic_loss,
                                                                    update_numbers=self._update_numbers,
                                                                    actor_loss_history=self._actor_loss_history,
                                                                    critic_loss_history=self._critic_loss_history)

    def _display_save_checkpoint_logger_statements(self, file_path: Path, start_time: datetime) -> None:
        self._logger.info("=" * 100)
        now = datetime.now(ZoneInfo("America/New_York"))
        formatted_start_time = start_time.strftime("%b-%d, %I:%M:%S %p")
        formatted_current_time = now.strftime("%b-%d, %I:%M:%S %p")
        self._logger.info(f"Start Time: {formatted_start_time}")
        self._logger.info(f"Current Time: {formatted_current_time}")
        self._logger.info(f"Successfully saved: {file_path}")
        self._logger.info("=" * 100)

    def _get_file_path(self, file_path_str: str, current_training_iteration: int, file_type_str: str) -> Path:

        model_weights_directory_path: Path = self._get_directory_path()

        model_weights_directory_path.mkdir(parents=True, exist_ok=True)

        filename_str: str = f"{file_path_str}_{current_training_iteration}_{self._timestamp_string}.{file_type_str}"
        checkpoint_path: Path = model_weights_directory_path / filename_str

        return checkpoint_path

    def _get_directory_path(self) -> Path:

        directory_path: Path = Path("model_weights")

        if isinstance(self._environment_obj, WareHouseEnv):
            directory_path = Path("model_weights/warehouse_env_files_1")

        elif isinstance(self._environment_obj, WareHouseEnv2):
            directory_path = Path("model_weights/warehouse_env_files_2")

        elif isinstance(self._environment_obj, WareHouseEnv3):
            directory_path = Path("model_weights/warehouse_env_files_2")

        return directory_path

    def _get_device(self):
        if torch.cuda.is_available():
            self._logger.info("Using Device CUDA")
            return torch.device("cuda")
        if torch.mps.is_available():
            self._logger.info("Using Device MPS")
            return torch.device("mps")

        self._logger.info("Using Device CPU")

        return torch.device("cpu")
