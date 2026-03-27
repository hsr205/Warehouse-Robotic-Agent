from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from gymnasium import Env
from torch import Tensor
from torch.optim import Adam
from tqdm import tqdm

from logger.logger import AppLogger
from models.actor_network import ActorNetwork
from models.critic_network import CriticNetwork
from warehouse_env.warehouse_env import WareHouseEnv


class WareHouseAgentPPOEvaluation:

    def __init__(self) -> None:
        self._device = self._get_device()
        self._learning_rate: float = 0.005
        self._environment_obj: Env = WareHouseEnv(render_mode=None)
        self._environment_obj_human_render_mode: Env = WareHouseEnv(render_mode='human')
        self._action_dimensions = self._environment_obj.action_space.n
        self._observation_dimensions = self._environment_obj.observation_space.get("direction").n

        self._actor_network: ActorNetwork = ActorNetwork(output_dimensions=self._action_dimensions,
                                                         device=self._device)
        self._critic_network: CriticNetwork = CriticNetwork(device=self._device)

        self._actor_network_optimizer: Adam = Adam(params=self._actor_network.parameters(),
                                                   lr=self._learning_rate)

        self._critic_network_optimizer: Adam = Adam(params=self._critic_network.parameters(),
                                                    lr=self._learning_rate)

        self._logger = AppLogger.get_logger(self.__class__.__name__)

    def evaluate_agent(self, num_episodes: int = 10) -> dict[str, int | float | list]:

        # TODO: Make the following more dynamic after testing
        checkpoint_path: Path = Path("")

        checkpoint_time_step: int = self._load_checkpoint(checkpoint_path=checkpoint_path)

        episode_returns_list: list[float] = []
        episode_lengths_list: list[int] = []

        for _ in tqdm(range(0, num_episodes), desc="Evaluate Agent Behaviour"):
            # observation_dict, info_dict = self._environment_obj.reset()

            # TODO: Remove after testing
            observation_dict, info_dict = self._environment_obj_human_render_mode.reset()

            is_terminated: bool = False
            is_truncated: bool = False
            episode_return: float = 0.0
            episode_length: int = 0
            is_done: bool = is_terminated or is_truncated

            while not is_done:
                action_int: int = self._get_evaluation_action(
                    observation_dict=observation_dict,
                )

                # TODO: Remove after testing
                # observation_dict, reward, is_terminated, is_truncated, info_dict = self._environment_obj.step(
                #     action_int
                # )

                observation_dict, reward, is_terminated, is_truncated, info_dict = self._environment_obj_human_render_mode.step(
                    action_int
                )

                episode_return += reward
                episode_length += 1

            episode_returns_list.append(episode_return)
            episode_lengths_list.append(episode_length)

        results_dict: dict[str, int | float | list] = {
            "checkpoint_path": checkpoint_path,
            "time_step": checkpoint_time_step,
            "num_episodes": num_episodes,
            "mean_return": float(np.mean(episode_returns_list)),
            "min_return": float(np.min(episode_lengths_list)),
            "max_return": float(np.max(episode_returns_list)),
            "mean_episode_length": float(np.mean(episode_lengths_list)),
            "std_episode_length": float(np.std(episode_lengths_list)),
            "episode_returns": episode_returns_list,
            "episode_lengths": episode_lengths_list
        }

        return results_dict

    def _get_evaluation_action(self, observation_dict: dict) -> int:

        observation_image_array: np.ndarray = observation_dict.get('image')

        observation_tensor: Tensor = torch.tensor(data=observation_image_array,
                                                  dtype=torch.float32,
                                                  device=self._device)

        if observation_tensor.dim() == 1:
            observation_tensor = observation_tensor.unsqueeze(0)

        with torch.no_grad():
            action_probabilities_tensor: Tensor = self._actor_network(observation_tensor)

            # NOTE: Always acting greedy, choosing the action with the highest probability from the softmax
            action_tensor: Tensor = torch.argmax(action_probabilities_tensor, dim=-1)

        action_int: int = int(action_tensor.item())

        return action_int

    def _load_checkpoint(self, checkpoint_path: Path) -> int:
        checkpoint_dict: dict[str, int | OrderedDict | dict] = torch.load(checkpoint_path, map_location=self._device)

        self._logger.info("=" * 100)
        self._logger.info(f"Successfully loaded: {checkpoint_path}")
        self._logger.info("=" * 100)

        self._actor_network.load_state_dict(checkpoint_dict["actor_state_dict"])
        self._critic_network.load_state_dict(checkpoint_dict["critic_state_dict"])
        self._actor_network_optimizer.load_state_dict(checkpoint_dict["actor_optimizer_state_dict"])
        self._critic_network_optimizer.load_state_dict(checkpoint_dict["critic_optimizer_state_dict"])

        self._actor_network.to(self._device)
        self._critic_network.to(self._device)

        self._actor_network.eval()
        self._critic_network.eval()

        return checkpoint_dict["current_training_iteration"]

    def _get_device(self):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.mps.is_available():
            return torch.device("mps")

        return torch.device("cpu")
