from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from sb3_contrib import TRPO
from stable_baselines3 import A2C, DQN
from stable_baselines3.common.monitor import Monitor

from logger.logger import AppLogger
from utils.training_history import TrainingHistory
from utils.warehouse_agent_baseline_rewards_callback import WarehouseAgentBaselineRewardsCallBack
from utils.warehouse_environment_adapter import WarehouseEnvironmentAdapter
from warehouse_env.warehouse_env import WareHouseEnv
from warehouse_env.warehouse_env_2 import WareHouseEnv2
from warehouse_env.warehouse_env_3 import WareHouseEnv3


class WareHouseAgentBaseline:

    def __init__(
            self,
            environment_obj: WareHouseEnv | WareHouseEnv2 | WareHouseEnv3,
            total_time_steps: int,
            algorithm_name: str,
            model_output_directory: Path | None = None,
    ) -> None:
        self._environment_obj = environment_obj
        self._total_time_steps = total_time_steps
        self._algorithm_name = algorithm_name.lower()
        self._model_output_directory = model_output_directory

        self._gamma: float = 0.95
        self._learning_rate: float = 3e-4

        self._logger = AppLogger.get_logger(self.__class__.__name__)
        self._warehouse_agent_baseline_reward_callback = WarehouseAgentBaselineRewardsCallBack(
            total_time_steps=self._total_time_steps,
            algorithm_name_str=self._algorithm_name.upper(),
        )

        self._timestamp_string: str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    def get_training_history(self) -> TrainingHistory:
        return TrainingHistory(
            algorithm_name=self._algorithm_name.upper(),
            environment_name=self._environment_obj.__class__.__name__,
            training_time_steps=self._warehouse_agent_baseline_reward_callback.training_time_steps,
            training_rewards=self._warehouse_agent_baseline_reward_callback.training_rewards,
            training_episode_numbers=self._warehouse_agent_baseline_reward_callback.training_episode_numbers,
            training_episode_rewards=self._warehouse_agent_baseline_reward_callback.training_episode_rewards,
            training_episode_time_steps=self._warehouse_agent_baseline_reward_callback.training_episode_time_steps,
        )

    def train_agent(self) -> None:
        start_time: datetime = datetime.now(ZoneInfo("America/New_York"))

        environment = Monitor(
            WarehouseEnvironmentAdapter(environment_obj=self._environment_obj)
        )

        model = self._create_model(environment=environment)

        self._logger.info("=" * 100)
        self._logger.info(f"Executing Algorithm Training: {self._algorithm_name.upper()}")
        self._logger.info(f"Environment: {self._environment_obj.__class__.__name__}")
        self._logger.info(f"Total Timesteps: {self._total_time_steps:,}")
        self._logger.info("=" * 100)

        model.learn(
            total_timesteps=self._total_time_steps,
            callback=self._warehouse_agent_baseline_reward_callback,
            progress_bar=False,
        )

        if self._model_output_directory is not None:
            model_save_file_path: Path = self._get_model_save_path()

            self._logger.info(f"Saving Model Elements To: {model_save_file_path}")
            model.save(str(model_save_file_path))
            self._logger.info(f"Successfully Saved Model Elements To: {model_save_file_path}")

        self._logger.info("=" * 100)
        self._logger.info(f"Completed Algorithm Training: {self._algorithm_name.upper()}")
        self._logger.info(f"Start Time: {start_time.strftime('%b-%d, %I:%M:%S %p')}")
        self._logger.info(
            f"End Time: {datetime.now(ZoneInfo('America/New_York')).strftime('%b-%d, %I:%M:%S %p')}"
        )
        self._logger.info("=" * 100)

        environment.close()

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

        raise ValueError(f"Unsupported algorithm name: {self._algorithm_name}")

    def _get_model_save_path(self) -> Path:
        if self._model_output_directory is None:
            raise ValueError("Model output directory must be configured before saving a baseline model.")

        self._model_output_directory.mkdir(parents=True, exist_ok=True)

        return self._model_output_directory / f"{self._algorithm_name}_{self._timestamp_string}"
