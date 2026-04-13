from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from gymnasium import Env
from gymnasium.spaces import Discrete
from torch import Tensor
from torch.optim import Adam
from tqdm import tqdm

from logger.logger import AppLogger
from models.actor_network import ActorNetwork
from models.critic_network import CriticNetwork
from utils.constants import Constants
from warehouse_env.warehouse_env_3_A2C import WareHouseEnv3A2C
from torch.distributions import Categorical

class WareHouseAgentA2CEvaluation:
    def __init__(self) -> None:
        self._learning_rate: float = 3e-4
        self._environment_obj: Env = WareHouseEnv3A2C(render_mode=None)
        self._environment_obj_human_render_mode: Env = WareHouseEnv3A2C(render_mode="human")
        self._logger = AppLogger.get_logger(self.__class__.__name__)

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

    def evaluate_agent(self, num_episodes: int = 10, render_human: bool = True) -> dict[str, int | float | list]:
        checkpoint_path: Path = Path("model_weights_a2c")
#####  Different than Henrys. Only uses the latest checkpoint
        if checkpoint_path.is_dir():
            checkpoint_files = sorted(checkpoint_path.glob("*.pt"), key=lambda p: p.stat().st_mtime)
            if len(checkpoint_files) == 0:
                raise FileNotFoundError("No A2C checkpoint files found in model_weights_a2c/")
            checkpoint_path = checkpoint_files[-1]

        checkpoint_time_step: int = self._load_checkpoint(checkpoint_path=checkpoint_path)

        episode_returns_list: list[float] = []
        episode_lengths_list: list[int] = []

        environment_obj = self._environment_obj_human_render_mode if render_human else self._environment_obj

        for _ in tqdm(range(num_episodes), desc="Evaluate A2C Agent Behaviour"):
            observation_dict, info_dict = environment_obj.reset()

            is_terminated: bool = False
            is_truncated: bool = False
            episode_return: float = 0.0
            episode_length: int = 0

            while not (is_terminated or is_truncated):
                action_int: int = self._get_evaluation_action(observation_dict=observation_dict)

                self._logger.info("=" * 100)
                self._logger.info(
                    f"Evaluation Action taken: {Constants.ACTION_SPACE_MAPPING_DICT.get(action_int, '')}"
                )
                self._logger.info("=" * 100)

                observation_dict, reward, is_terminated, is_truncated, info_dict = environment_obj.step(action_int)
                self._logger.info(f"Agent Pos: {environment_obj.agent_pos}, Agent Dir: {environment_obj.agent_dir}")

                self._logger.info(
                    f"Reward: {reward} -> "
                    f"Is Terminated: {is_terminated} -> "
                    f"Is Truncated: {is_truncated} -> "
                    f"Info Dict: {info_dict}"
                )

                episode_return += reward
                episode_length += 1

            episode_returns_list.append(episode_return)
            episode_lengths_list.append(episode_length)

        return {
            "checkpoint_path": str(checkpoint_path),
            "time_step": checkpoint_time_step,
            "num_episodes": num_episodes,
            "mean_return": float(np.mean(episode_returns_list)),
            "min_return": float(np.min(episode_returns_list)),
            "max_return": float(np.max(episode_returns_list)),
            "mean_episode_length": float(np.mean(episode_lengths_list)),
            "std_episode_length": float(np.std(episode_lengths_list)),
            "episode_returns": episode_returns_list,
            "episode_lengths": episode_lengths_list,
        }
    
    # #Henry approach 
    # def _get_all_checkpoint_file_paths_list(self) -> list[Path]:
    #     checkpoint_dir: Path = Path("model_weights_a2c")

    #     checkpoint_paths_list: list[Path] = []

    #     for file_path in checkpoint_dir.iterdir():
    #         if file_path.is_file() and file_path.suffix == ".pt":
    #             checkpoint_paths_list.append(file_path)

    #     checkpoint_paths_list.sort(key=lambda p: p.stat().st_mtime)

    #     return checkpoint_paths_list
    # ##### End of henry approach

    def _get_evaluation_action(self, observation_dict: dict) -> int:
        observation_image_array: np.ndarray = observation_dict.get("image")

        observation_tensor: Tensor = torch.tensor(
            data=observation_image_array,
            dtype=torch.float32,
            device=self._device
        )
        # with torch.no_grad():
        #     action_probabilities_tensor: Tensor = self._actor_network(observation_tensor)

        #     self._logger.info(f"action_probabilities_tensor = {action_probabilities_tensor}")

        #     categorical_distribution: Categorical = Categorical(probs=action_probabilities_tensor)
        #     action_tensor: Tensor = categorical_distribution.sample()
            
        with torch.no_grad():
            action_probabilities_tensor: Tensor = self._actor_network(observation_tensor)

            self._logger.info(f"action_probabilities_tensor = {action_probabilities_tensor}")

            action_tensor: Tensor = torch.argmax(action_probabilities_tensor, dim=-1)

        return int(action_tensor.item())

    def _load_checkpoint(self, checkpoint_path: Path) -> int:
        checkpoint_dict: dict[str, int | OrderedDict | dict] = torch.load(
            checkpoint_path,
            map_location=self._device
        )

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

        return int(checkpoint_dict["current_training_iteration"])

    def _get_device(self):
        if torch.cuda.is_available():
            self._logger.info("Using Device CUDA")
            return torch.device("cuda")
        if torch.mps.is_available():
            self._logger.info("Using Device MPS")
            return torch.device("mps")

        self._logger.info("Using Device CPU")
        return torch.device("cpu")
