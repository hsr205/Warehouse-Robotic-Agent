from gymnasium import Env
import time
from logger.logger import AppLogger
from warehouse_env.warehouse_env import WareHouseEnv


class WareHouseAgentPPO:

    def __init__(self) -> None:
        self._environment_obj: Env = WareHouseEnv(render_mode="human")
        self._logger = AppLogger.get_logger(self.__class__.__name__)


    def test_method(self) -> None:

        observation_dict, info_dict = self._environment_obj.reset(seed=42)

        for _ in range(300):
            action_int = self._environment_obj.action_space.sample()
            observation_dict, reward_float, terminated_bool, truncated_bool, info_dict = self._environment_obj.step(
                action_int
            )

            self._logger.info(
                f"reward={reward_float:.2f}, "
                f"terminated={terminated_bool}, "
                f"truncated={truncated_bool}, "
                f"info={info_dict}"
            )

            time.sleep(0.15)

            if terminated_bool or truncated_bool:
                observation_dict, info_dict = self._environment_obj.reset()

        self._environment_obj.close()

    def get_proximal_policy(self, observation_dict: dict) -> int:
        return 0
