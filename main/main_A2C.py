from logging import Logger

from logger.logger import AppLogger
from models.warehouse_agent_a2c import WareHouseAgentA2C
from models.warehouse_agent_a2c_evaluation import WareHouseAgentA2CEvaluation


def main() -> int:
    logger: Logger = AppLogger().get_logger(__name__)

    warehouse_agent_a2c = WareHouseAgentA2C()
    warehouse_agent_a2c_evaluation = WareHouseAgentA2CEvaluation()

    try:
        # warehouse_agent_a2c.train_agent()
        results_dict = warehouse_agent_a2c_evaluation.evaluate_agent(
            num_episodes=1,
            render_human= True
        )
        logger.info(f"A2C Evaluation Results: {results_dict}")
        return 0

    except Exception as e:
        logger.error(f"Exception Thrown: {e}")

    return 1


if __name__ == "__main__":
    main()