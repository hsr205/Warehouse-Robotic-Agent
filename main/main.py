from logging import Logger

from logger.logger import AppLogger
from models.warehouse_agent_ppo import WareHouseAgentPPO
from models.warehouse_agent_ppo_evaluation import WareHouseAgentPPOEvaluation
from warehouse_env.warehouse_env import WareHouseEnv


def main() -> int:
    logger: Logger = AppLogger().get_logger(__name__)
    warehouse_agent_ppo: WareHouseAgentPPO = WareHouseAgentPPO()
    warehouse_agent_ppo_evaluation: WareHouseAgentPPOEvaluation = WareHouseAgentPPOEvaluation()
    warehouse_env: WareHouseEnv = WareHouseEnv()

    try:

        warehouse_env.randomly_navigate_custom_grid_world()
        # warehouse_agent_ppo.train_agent()
        # warehouse_agent_ppo_evaluation.evaluate_agent()

        return 0

    except Exception as e:
        logger.error(f"Exception Thrown: {e}")

    return 1


if __name__ == "__main__":
    main()
