from logging import Logger

from logger.logger import AppLogger
from models.warehouse_agent_ppo import WareHouseAgentPPO
from models.warehouse_agent_ppo_evaluation import WareHouseAgentPPOEvaluation
from warehouse_env.warehouse_env import WareHouseEnv
from warehouse_env.warehouse_env_2 import WareHouseEnv2
from warehouse_env.warehouse_env_3 import WareHouseEnv3


def main() -> int:
    logger: Logger = AppLogger().get_logger(__name__)

    warehouse_env: WareHouseEnv = WareHouseEnv(render_mode=None)
    warehouse_env_2: WareHouseEnv2 = WareHouseEnv2(render_mode="human")
    warehouse_env_3: WareHouseEnv3 = WareHouseEnv3(render_mode=None)

    warehouse_agent_ppo_warehouse_env: WareHouseAgentPPO = WareHouseAgentPPO(environment_obj=warehouse_env,
                                                                             total_actions_taken_during_training_episode=2_000,
                                                                             batch_size_before_policy_update=4_000)

    warehouse_agent_ppo_warehouse_env_2: WareHouseAgentPPO = WareHouseAgentPPO(environment_obj=warehouse_env_2,
                                                                               total_actions_taken_during_training_episode=3_000,
                                                                               batch_size_before_policy_update=6_000)

    warehouse_agent_ppo_warehouse_env_3: WareHouseAgentPPO = WareHouseAgentPPO(environment_obj=warehouse_env_3,
                                                                               total_actions_taken_during_training_episode=3_000,
                                                                               batch_size_before_policy_update=6_000)

    warehouse_agent_ppo_evaluation_warehouse_env: WareHouseAgentPPOEvaluation = WareHouseAgentPPOEvaluation(
        environment_obj=warehouse_env)
    warehouse_agent_ppo_evaluation_warehouse_env_2: WareHouseAgentPPOEvaluation = WareHouseAgentPPOEvaluation(
        environment_obj=warehouse_env_2)
    warehouse_agent_ppo_evaluation_warehouse_env_3: WareHouseAgentPPOEvaluation = WareHouseAgentPPOEvaluation(
        environment_obj=warehouse_env_3)

    try:

        # warehouse_env.randomly_navigate_custom_grid_world()
        # warehouse_env_2.randomly_navigate_custom_grid_world()
        # warehouse_env_3.randomly_navigate_custom_grid_world()

        # warehouse_agent_ppo_warehouse_env.train_agent()
        # warehouse_agent_ppo_warehouse_env_2.train_agent()
        warehouse_agent_ppo_warehouse_env_3.train_agent()

        # warehouse_agent_ppo_evaluation_warehouse_env.evaluate_agent()
        # warehouse_agent_ppo_evaluation_warehouse_env_2.evaluate_agent()
        # warehouse_agent_ppo_evaluation_warehouse_env_3.evaluate_agent()

        return 0

    except Exception as e:
        logger.error(f"Exception Thrown: {e}")

    return 1


if __name__ == "__main__":
    main()
