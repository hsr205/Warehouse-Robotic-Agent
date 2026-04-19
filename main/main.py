from __future__ import annotations

from logging import Logger
from pathlib import Path

from logger.logger import AppLogger
from models.warehouse_agent_a3c import WareHouseAgentA3C
from models.warehouse_agent_baseline import WareHouseAgentBaseline
from utils.model_plotting import ModelPlotting
from utils.training_history import TrainingHistory
from warehouse_env.warehouse_env import WareHouseEnv
from warehouse_env.warehouse_env_2 import WareHouseEnv2
from warehouse_env.warehouse_env_3 import WareHouseEnv3


ENVIRONMENT_CLASSES: list[type[WareHouseEnv] | type[WareHouseEnv2] | type[WareHouseEnv3]] = [
    WareHouseEnv,
    WareHouseEnv2,
    WareHouseEnv3,
]

BASELINE_ALGORITHM_NAMES: list[str] = ["a2c", "dqn", "trpo"]
COMPARISON_SNAPSHOT_TIMESTEP_INTERVAL: int = 500_000

BASELINE_TOTAL_TIME_STEPS_BY_ENVIRONMENT_NAME: dict[str, int] = {
    "WareHouseEnv": 2_000_000, #5_000_000,
   # "WareHouseEnv2": 10_000, #5_000_000,
    #"WareHouseEnv3": 10_000, #5_000_000,
}

A3C_MAX_GLOBAL_TIME_STEPS_BY_ENVIRONMENT_NAME: dict[str, int] = {
    "WareHouseEnv": 2_000_000, # 5_000_000,
   # "WareHouseEnv2": 10_000, # 5_000_000,
   # "WareHouseEnv3": 10_000, # 5_000_000,
}


def main() -> int:
    logger: Logger = AppLogger().get_logger(__name__)
    model_plotting: ModelPlotting = ModelPlotting()

    try:
        for environment_class in ENVIRONMENT_CLASSES:
            logger.info("=" * 100)
            logger.info(f"Starting comparison training for environment: {environment_class.__name__}")
            logger.info("=" * 100)

            training_histories_dict = train_all_algorithms_for_environment(environment_class=environment_class)
            environment_obj = environment_class(render_mode=None)

            try:
                model_plotting.create_comparison_plots_for_environment(
                    environment_obj=environment_obj,
                    training_histories_dict=training_histories_dict,
                )
                model_plotting.create_timestep_snapshot_plots_for_environment(
                    environment_obj=environment_obj,
                    training_histories_dict=training_histories_dict,
                    time_step_interval=COMPARISON_SNAPSHOT_TIMESTEP_INTERVAL,
                )
                model_plotting.create_episode_snapshot_plots_for_environment(
                    environment_obj=environment_obj,
                    training_histories_dict=training_histories_dict,
                    time_step_interval=COMPARISON_SNAPSHOT_TIMESTEP_INTERVAL,
                )
            finally:
                environment_obj.close()

        return 0

    except Exception as exception:
        logger.error(f"Exception Thrown: {exception}")
        return 1


def train_all_algorithms_for_environment(
        environment_class: type[WareHouseEnv] | type[WareHouseEnv2] | type[WareHouseEnv3],
) -> dict[str, TrainingHistory]:
    training_histories_dict: dict[str, TrainingHistory] = {}
    environment_name: str = environment_class.__name__

    baseline_total_time_steps: int = BASELINE_TOTAL_TIME_STEPS_BY_ENVIRONMENT_NAME[environment_name]

    baseline_model_output_directory = get_output_root_directory() / "baseline_models" / get_environment_directory_name(
        environment_class=environment_class
    )

    for baseline_algorithm_name in BASELINE_ALGORITHM_NAMES:
        baseline_environment_obj = environment_class(render_mode=None)

        baseline_agent = WareHouseAgentBaseline(
            environment_obj=baseline_environment_obj,
            total_time_steps=baseline_total_time_steps,
            algorithm_name=baseline_algorithm_name,
            model_output_directory=baseline_model_output_directory,
        )

        baseline_agent.train_agent()
        baseline_environment_obj.close()

        baseline_training_history: TrainingHistory = baseline_agent.get_training_history()
        training_histories_dict[baseline_training_history.algorithm_name] = baseline_training_history

    a3c_agent = WareHouseAgentA3C(
        environment_class=environment_class,
        number_of_workers=3,
        worker_rollout_steps=32,
        max_global_time_steps=A3C_MAX_GLOBAL_TIME_STEPS_BY_ENVIRONMENT_NAME[environment_name],
        max_time_steps_per_episode=500,
        checkpoint_interval_updates=10_000,
        resume_from_latest_checkpoint=False,
        checkpoint_directory_name=str(
            get_output_root_directory() / "a3c_checkpoints" / get_environment_directory_name(
                environment_class=environment_class
            )
        ),
        plot_directory_name=str(
            get_output_root_directory() / "a3c_training_plots" / get_environment_directory_name(
                environment_class=environment_class
            )
        ),
        base_seed=42,
    )

    a3c_agent.train_agent()
    a3c_training_history: TrainingHistory = a3c_agent.get_training_history()
    training_histories_dict[a3c_training_history.algorithm_name] = a3c_training_history

    return training_histories_dict


def save_training_histories_for_environment(
        environment_class: type[WareHouseEnv] | type[WareHouseEnv2] | type[WareHouseEnv3],
        training_histories_dict: dict[str, TrainingHistory],
) -> None:
    history_directory_path = get_history_directory_path(environment_class=environment_class)
    history_directory_path.mkdir(parents=True, exist_ok=True)

    for training_history in training_histories_dict.values():
        training_history.save_to_json(
            history_directory_path / f"{training_history.algorithm_name.lower()}_history.json",
            compact_time_step_history=True,
            max_time_step_points=25_000,
        )


def load_training_histories_for_environment(
        environment_class: type[WareHouseEnv] | type[WareHouseEnv2] | type[WareHouseEnv3],
) -> dict[str, TrainingHistory]:
    history_directory_path = get_history_directory_path(environment_class=environment_class)

    training_histories_dict: dict[str, TrainingHistory] = {}
    preferred_algorithm_order_list: list[str] = ["a2c", "dqn", "trpo", "a3c", "ppo"]

    for algorithm_name in preferred_algorithm_order_list:
        file_path = history_directory_path / f"{algorithm_name}_history.json"
        if file_path.exists():
            training_history = TrainingHistory.load_from_json(file_path=file_path)
            training_histories_dict[training_history.algorithm_name] = training_history

    for file_path in sorted(history_directory_path.glob("*_history.json")):
        training_history = TrainingHistory.load_from_json(file_path=file_path)
        if training_history.algorithm_name not in training_histories_dict:
            training_histories_dict[training_history.algorithm_name] = training_history

    return training_histories_dict


def get_history_directory_path(
        environment_class: type[WareHouseEnv] | type[WareHouseEnv2] | type[WareHouseEnv3],
) -> Path:
    return get_output_root_directory() / "training_histories" / get_environment_directory_name(
        environment_class=environment_class
    )


def get_output_root_directory() -> Path:
    return Path("model_outputs")


def get_environment_directory_name(
        environment_class: type[WareHouseEnv] | type[WareHouseEnv2] | type[WareHouseEnv3],
) -> str:
    if environment_class is WareHouseEnv:
        return "warehouse_env_files_1"

    if environment_class is WareHouseEnv2:
        return "warehouse_env_files_2"

    if environment_class is WareHouseEnv3:
        return "warehouse_env_files_3"

    raise ValueError(f"Unsupported environment type: {environment_class.__name__}")


if __name__ == "__main__":
    raise SystemExit(main())