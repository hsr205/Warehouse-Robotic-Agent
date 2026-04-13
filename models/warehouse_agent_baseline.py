from __future__ import annotations
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from sb3_contrib import TRPO
from stable_baselines3 import A2C, DQN
from stable_baselines3.common.monitor import Monitor

from logger.logger import AppLogger
from utils.training_history import TrainingHistory
from utils.model_plotting import ModelPlotting
from utils.warehouse_agent_baseline_rewards_callback import WarehouseAgentBaselineRewardsCallBack
from utils.warehouse_environment_adapter import WarehouseEnvironmentAdapter
from warehouse_env.warehouse_env import WareHouseEnv
from warehouse_env.warehouse_env_2 import WareHouseEnv2
from warehouse_env.warehouse_env_3 import WareHouseEnv3


class WareHouseAgentBaseline:

    def __init__(self, environment_obj: WareHouseEnv | WareHouseEnv2 | WareHouseEnv3, total_time_steps: int,
                 algorithm_name: str) -> None:
        self._environment_obj = environment_obj
        self._total_time_steps = total_time_steps
        self._algorithm_name = algorithm_name.lower()

        self._gamma: float = 0.95
        self._learning_rate: float = 3e-4

        self._model_plotting: ModelPlotting = ModelPlotting()
        self._logger = AppLogger.get_logger(self.__class__.__name__)
        self._warehouse_agent_baseline_reward_callback = WarehouseAgentBaselineRewardsCallBack(
            total_time_steps=self._total_time_steps,
            algorithm_name_str=self._algorithm_name.upper()
        )

        self._timestamp_string: str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self._trained_model_file_path: Path | None = None

    def get_training_history(self) -> TrainingHistory:
        return TrainingHistory(
            algorithm_name=self._algorithm_name.upper(),
            training_time_steps=self._warehouse_agent_baseline_reward_callback.training_time_steps,
            training_rewards=self._warehouse_agent_baseline_reward_callback.training_rewards,
            training_episode_numbers=self._warehouse_agent_baseline_reward_callback.training_episode_numbers,
            training_episode_rewards=self._warehouse_agent_baseline_reward_callback.training_episode_rewards,
        )

    def train_agent(self) -> None:
        start_time: datetime = datetime.now(ZoneInfo("America/New_York"))

        environment = Monitor(
            WarehouseEnvironmentAdapter(environment_obj=self._environment_obj)
        )

        model = self._create_model(environment=environment)

        self._logger.info("=" * 100)
        self._logger.info(f"Executing Algorithm Training: {self._algorithm_name.upper()}")
        self._logger.info(f"Total Timesteps: {self._total_time_steps:,}")
        self._logger.info("=" * 100)

        model.learn(
            total_timesteps=self._total_time_steps,
            callback=self._warehouse_agent_baseline_reward_callback,
            progress_bar=False,
        )

        model_save_file_path: Path = self._get_model_save_path()
        model.save(str(model_save_file_path))

        self._trained_model_file_path = Path(f"{model_save_file_path}.zip")

        self._logger.info("=" * 100)
        self._logger.info(f"Completed Algorithm Training: {self._algorithm_name.upper()}")
        self._logger.info(f"Saving Model To: {model_save_file_path}")
        self._logger.info(f"Start Time: {start_time.strftime('%b-%d, %I:%M:%S %p')}")
        self._logger.info(
            f"End Time: {datetime.now(ZoneInfo('America/New_York')).strftime('%b-%d, %I:%M:%S %p')}"
        )
        self._logger.info("=" * 100)

        environment.close()

    def evaluate_agent(self, num_episodes: int = 20, deterministic: bool = True) -> dict[str, float]:
        environment_obj: WarehouseEnvironmentAdapter = WarehouseEnvironmentAdapter(
            environment_obj=self._environment_obj)
        model = self._load_model(environment=environment_obj)

        episode_rewards: list[float] = []

        for episode_index_num in range(0, num_episodes):
            observation_array, _ = environment_obj.reset(seed=1000 + episode_index_num)
            is_done: bool = False
            total_reward: float = 0.0

            while not is_done:
                action, _ = model.predict(
                    observation_array,
                    deterministic=deterministic,
                )

                observation_array, reward, terminated, truncated, _ = environment_obj.step(int(action))
                is_done = bool(terminated or truncated)

                total_reward += float(reward)

            episode_rewards.append(total_reward)

        environment_obj.close()

        average_reward: float = float(np.mean(episode_rewards))
        standard_deviation_reward: float = float(np.std(episode_rewards))

        results_dict: dict[str, float] = {
            "mean_reward": average_reward,
            "std_reward": standard_deviation_reward,
            "num_episodes": float(num_episodes),
        }

        return results_dict

    def _create_model(self, environment: Monitor):

        if self._algorithm_name == "a2c":
            return A2C(
                policy="MlpPolicy",
                env=environment,
                learning_rate=self._learning_rate,
                gamma=self._gamma,
                n_steps=5,
                ent_coef=0.075,
                verbose=0,
                device="auto",
            )

        if self._algorithm_name == "trpo":
            return TRPO(
                policy="MlpPolicy",
                env=environment,
                learning_rate=self._learning_rate,
                gamma=self._gamma,
                batch_size=64,
                verbose=0,
                device="auto",
            )

        if self._algorithm_name == "dqn":
            return DQN(
                policy="MlpPolicy",
                env=environment,
                learning_rate=1e-4,
                gamma=self._gamma,
                buffer_size=100_000,
                learning_starts=5_000,
                batch_size=64,
                train_freq=4,
                target_update_interval=1_000,
                verbose=0,
                device="auto",
            )

    def _get_plot_file_path(self, file_name_prefix: str) -> Path:
        directory_path: Path = self._get_directory_path()
        directory_path.mkdir(parents=True, exist_ok=True)

        file_name: str = f"{file_name_prefix}_{self._algorithm_name}_{self._timestamp_string}.png"

        return directory_path / file_name

    def _load_model(self, environment: WarehouseEnvironmentAdapter):

        model_path: Path = self._trained_model_file_path

        if not model_path.exists():
            raise FileNotFoundError(f"Saved model file does not exist: {model_path}")

        if self._algorithm_name == "a2c":
            return A2C.load(str(model_path), env=environment, device="auto")

        if self._algorithm_name == "trpo":
            return TRPO.load(str(model_path), env=environment, device="auto")

        if self._algorithm_name == "dqn":
            return DQN.load(str(model_path), env=environment, device="auto")

        raise ValueError(f"Unsupported algorithm name: {self._algorithm_name}")

    def _plot_rewards_by_time_step(self) -> None:
        file_path: Path = self._get_plot_file_path(file_name_prefix="baseline_rewards_by_time_step")

        self._model_plotting.plot_rewards_by_time_step(
            file_path=file_path,
            training_time_steps=self._warehouse_agent_baseline_reward_callback.training_time_steps,
            training_rewards=self._warehouse_agent_baseline_reward_callback.training_rewards,
        )

    def _plot_rewards_by_episode(self) -> None:
        file_path: Path = self._get_plot_file_path(file_name_prefix="baseline_rewards_by_episode")

        self._model_plotting.plot_rewards_by_episode(
            file_path=file_path,
            training_episode_numbers=self._warehouse_agent_baseline_reward_callback.training_episode_numbers,
            training_episode_rewards=self._warehouse_agent_baseline_reward_callback.training_episode_rewards,
        )

    def _get_model_save_path(self) -> Path:
        directory_path: Path = self._get_directory_path()
        directory_path.mkdir(parents=True, exist_ok=True)

        if not directory_path.exists():
            raise FileNotFoundError(f"Directory does not exist: {directory_path}")

        timestamp_string: str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

        model_save_file_path: Path = directory_path / f"{self._algorithm_name}_{timestamp_string}"

        self._logger.info("=" * 100)
        self._logger.info(f"Resolved model save path: {model_save_file_path}")
        self._logger.info("=" * 100)

        return model_save_file_path

    def _get_directory_path(self) -> Path:
        project_root: Path = Path(__file__).resolve().parents[1]

        directory_path: Path = project_root / "model_weights"

        if isinstance(self._environment_obj, WareHouseEnv):
            directory_path = project_root / "model_weights" / "baseline_model_outputs" / "warehouse_env_files_1"

        elif isinstance(self._environment_obj, WareHouseEnv2):
            directory_path = project_root / "model_weights" / "baseline_model_outputs" / "warehouse_env_files_2"

        elif isinstance(self._environment_obj, WareHouseEnv3):
            directory_path = project_root / "model_weights" / "baseline_model_outputs" / "warehouse_env_files_3"

        self._logger.info("=" * 100)
        self._logger.info(f"Resolved directory path: {directory_path}")
        self._logger.info("=" * 100)

        return directory_path
