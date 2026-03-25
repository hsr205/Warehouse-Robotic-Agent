from gymnasium import Env
import time
from logger.logger import AppLogger
from utils.constants import Constants
from warehouse_env.warehouse_env import WareHouseEnv


class WareHouseAgentPPO:

    def __init__(self) -> None:
        self._environment_obj: Env = WareHouseEnv(render_mode=None)
        self._logger = AppLogger.get_logger(self.__class__.__name__)


    def simulate_agent_environment_navigation(self) -> None:

        observation_dict, info_dict = self._environment_obj.reset(seed=42)



        rewards_list: list[float] = []
        action_taken_list: list[int] = []

        for _ in range(300):
            action_int = self._environment_obj.action_space.sample()
            observation_dict, reward, terminated_bool, truncated_bool, info_dict = self._environment_obj.step(
                action_int
            )

            action_str:str = Constants.ACTION_SPACE_MAPPING_DICT.get(action_int, "")

            self._logger.info(
                f"action_int={action_int}, "
                f"reward={reward:.2f}, "
                f"action_str={action_str}, "
                f"terminated={terminated_bool}, "
                f"truncated={truncated_bool}, "
                f"info={info_dict}"
            )

            reward_float: float = float(reward)
            rewards_list.append(reward_float)
            action_taken_list.append(action_int)

            time.sleep(0.15)

            if terminated_bool or truncated_bool:
                observation_dict, info_dict = self._environment_obj.reset()

        self._environment_obj.close()

    def get_proximal_policy(self, observation_dict: dict) -> int:
        return 0
