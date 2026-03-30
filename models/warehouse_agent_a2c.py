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
from warehouse_env.warehouse_env import WareHouseEnv


class WareHouseAgentA2C:
    def __init__(self) -> None:
        self._gamma: float = 0.95
        self._learning_rate: float = 3e-4
        self._entropy_coefficient: float = 0.01
        self._critic_coefficient: float = 0.5

        self._total_training_iterations: int = 100
        self._time_steps_per_batch: int = 1000
        self._max_time_steps_per_episode: int = 2000

        self._environment_obj: Env = WareHouseEnv(render_mode=None)
        self._logger = AppLogger.get_logger(self.__class__.__name__)

        # Keep same 3-action setup as PPO for fair comparison
        # 0 = LEFT, 1 = RIGHT, 2 = FORWARD
        self._action_dimensions: int = Discrete(3).n

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

    def train_agent(self) -> None:
        progress_bar: tqdm = tqdm(
            total=self._total_training_iterations,
            desc="Training Warehouse A2C Agent"
        )

        for current_training_iteration in range(self._total_training_iterations + 1):

            if current_training_iteration > 0 and current_training_iteration % 50 == 0:
                self._save_checkpoint(current_training_iteration=current_training_iteration)

            (
                batch_observation_tensor,
                batch_actions_tensor,
                batch_returns_tensor,
            ) = self._rollout()

            state_values_tensor: Tensor = self._critic_network(batch_observation_tensor).squeeze(-1)

            advantage_tensor: Tensor = batch_returns_tensor - state_values_tensor.detach()
            advantage_tensor = self._normalize_tensor(advantage_tensor)

            action_probabilities_tensor: Tensor = self._actor_network(batch_observation_tensor)
            categorical_distribution: Categorical = Categorical(probs=action_probabilities_tensor)

            log_probabilities_tensor: Tensor = categorical_distribution.log_prob(batch_actions_tensor)
            entropy_tensor: Tensor = categorical_distribution.entropy().mean()

            actor_loss_tensor: Tensor = -(log_probabilities_tensor * advantage_tensor).mean()
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

            mean_return = float(batch_returns_tensor.mean().item())
            mean_value = float(state_values_tensor.mean().item())

            self._logger.info("=" * 100)
            self._logger.info(
                f"[A2C] Iteration={current_training_iteration} | "
                f"Actor Loss={actor_loss_tensor.item():.6f} | "
                f"Critic Loss={critic_loss_tensor.item():.6f} | "
                f"Entropy={entropy_tensor.item():.6f} | "
                f"Mean Return={mean_return:.6f} | "
                f"Mean Value={mean_value:.6f}"
            )
            self._logger.info("=" * 100)

            progress_bar.update(1)

        progress_bar.close()
        self._environment_obj.close()

    def _rollout(self) -> tuple[Tensor, Tensor, Tensor]:
        current_time_step: int = 0

        batch_observation_list: list[dict] = []
        batch_actions_list: list[int] = []
        batch_rewards_list: list[list[float]] = []

        while current_time_step < self._time_steps_per_batch:
            episode_rewards_list: list[float] = []
            observation_dict, info_dict = self._environment_obj.reset(seed=42 + current_time_step)

            for _ in range(self._max_time_steps_per_episode):
                if current_time_step >= self._time_steps_per_batch:
                    break

                batch_observation_list.append(observation_dict)

                action_int: int = self._get_action(observation_dict=observation_dict)

                observation_dict, reward, is_terminated, is_truncated, info_dict = self._environment_obj.step(
                    action_int
                )

                batch_actions_list.append(action_int)
                episode_rewards_list.append(float(reward))

                current_time_step += 1

                if is_terminated or is_truncated:
                    break

            batch_rewards_list.append(episode_rewards_list)

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

        return batch_observation_tensor, batch_actions_tensor, batch_returns_tensor

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

    def _get_file_path(self, current_training_iteration: int) -> Path:
        model_weights_directory_path: Path = Path("model_weights_a2c")
        model_weights_directory_path.mkdir(parents=True, exist_ok=True)

        timestamp_string: str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        filename: str = f"a2c_checkpoint_step_{current_training_iteration}_{timestamp_string}.pt"

        return model_weights_directory_path / filename

    def _get_device(self):
        if torch.cuda.is_available():
            self._logger.info("Using Device CUDA")
            return torch.device("cuda")
        if torch.mps.is_available():
            self._logger.info("Using Device MPS")
            return torch.device("mps")

        self._logger.info("Using Device CPU")
        return torch.device("cpu")