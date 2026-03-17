from logging import Logger

from logger.logger import AppLogger
from models.robotic_ppo import RoboticPPO
from warehouse_env.warehouse_env import WareHouseEnv


def main() -> int:
    logger: Logger = AppLogger().get_logger(__name__)
    robotic_ppo_obj: RoboticPPO = RoboticPPO()
    warehouse_env: WareHouseEnv = WareHouseEnv()

    try:

        warehouse_env.randomly_navigate_custom_grid_world()

        return 0

    except Exception as e:
        logger.error(f"Exception Thrown: {e}")

    return 1


if __name__ == "__main__":
    main()
