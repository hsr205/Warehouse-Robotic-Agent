from logging import Logger

from logger.logger import AppLogger
from models.warehouse_agent_baseline import WareHouseAgentBaseline
from models.warehouse_agent_ppo import WareHouseAgentPPO
from utils.model_plotting import ModelPlotting
from utils.training_history import TrainingHistory
from warehouse_env.warehouse_env import WareHouseEnv
from warehouse_env.warehouse_env_2 import WareHouseEnv2
from warehouse_env.warehouse_env_3 import WareHouseEnv3


def main() -> int:
    logger: Logger = AppLogger().get_logger(__name__)

    model_plotting: ModelPlotting = ModelPlotting()

    warehouse_env: WareHouseEnv = WareHouseEnv(render_mode=None)
    warehouse_env_2: WareHouseEnv2 = WareHouseEnv2(render_mode=None)
    warehouse_env_3: WareHouseEnv3 = WareHouseEnv3(render_mode=None)

    warehouse_env_list: list = [warehouse_env, warehouse_env_2, warehouse_env_3]

    try:
        for current_environment_obj in warehouse_env_list:
            training_histories_dict: dict[str, TrainingHistory] = train_all_algorithms_for_one_environment(
                environment_obj=current_environment_obj
            )

            model_plotting.create_comparison_plots_for_environment(
                environment_obj=current_environment_obj,
                training_histories_dict=training_histories_dict,
            )

        return 0

    except Exception as exception:
        logger.error(f"Exception Thrown: {exception}")
        return 1


def get_total_time_steps_for_environment(environment_obj) -> int:
    if isinstance(environment_obj, WareHouseEnv):
        return 4_000_000

    if isinstance(environment_obj, WareHouseEnv2):
        return 18_000_000

    if isinstance(environment_obj, WareHouseEnv3):
        return 18_000_000

    raise ValueError(f"Unsupported environment type: {type(environment_obj).__name__}")


def train_all_algorithms_for_one_environment(environment_obj) -> dict[str, TrainingHistory]:
    baseline_algorithms_list: list[str] = ["a2c", "dqn", "trpo"]
    training_histories_dict: dict[str, TrainingHistory] = {}

    total_time_steps: int = get_total_time_steps_for_environment(environment_obj)

    for baseline_algorithm_name in baseline_algorithms_list:
        baseline_agent: WareHouseAgentBaseline = WareHouseAgentBaseline(
            environment_obj=environment_obj,
            total_time_steps=total_time_steps,
            algorithm_name=baseline_algorithm_name,
        )

        baseline_agent.train_agent()
        baseline_training_history: TrainingHistory = baseline_agent.get_training_history()
        training_histories_dict[baseline_training_history.algorithm_name] = baseline_training_history

    if isinstance(environment_obj, WareHouseEnv):
        ppo_agent: WareHouseAgentPPO = WareHouseAgentPPO(
            environment_obj=environment_obj,
            total_actions_taken_during_training_episode=2_000,
            batch_size_before_policy_update=4_000,
        )
    else:
        ppo_agent = WareHouseAgentPPO(
            environment_obj=environment_obj,
            total_actions_taken_during_training_episode=3_000,
            batch_size_before_policy_update=6_000,
        )

    ppo_agent.train_agent()
    ppo_training_history: TrainingHistory = ppo_agent.get_training_history()
    training_histories_dict[ppo_training_history.algorithm_name] = ppo_training_history

    return training_histories_dict


if __name__ == "__main__":
    main()
