from dataclasses import dataclass

from gymnasium import Env

from logger.logger import AppLogger
from models.actor_network import ActorNetwork
from utils.constants import Constants
from warehouse_env.warehouse_env import WareHouseEnv


@dataclass
class StepDataObj:
    observation_dict: dict
    action_int: int
    action_str: str
    reward_float: float
    is_done: bool
    info_dict: dict


class WareHouseAgentPPO:

    def __init__(self) -> None:

        self._environment_obj: Env = WareHouseEnv(render_mode=None)
        self._action_dimensions = self._environment_obj.action_space.shape[0]
        self._observation_dimensions = self._environment_obj.observation_space.shape[0]
        self._actor_network:ActorNetwork = ActorNetwork(self._observation_dimensions, self._action_dimensions)
        self._logger = AppLogger.get_logger(self.__class__.__name__)

    def simulate_agent_environment_navigation(self, num_trial_runs: int) -> None:

        for num_trial in range(0, num_trial_runs):

            observation_dict, info_dict = self._environment_obj.reset(seed=42)

            is_done: bool = False
            episode_step_counter: int = 0
            rewards_list: list[float] = []
            action_taken_list: list[int] = []

            while not is_done:
                episode_step_counter += 1
                action_int = self._environment_obj.action_space.sample()
                observation_dict, reward, terminated_bool, truncated_bool, info_dict = self._environment_obj.step(
                    action_int
                )

                action_str: str = Constants.ACTION_SPACE_MAPPING_DICT.get(action_int, "")
                is_done = terminated_bool or truncated_bool
                reward_float: float = float(reward)

                step_data_obj: StepDataObj = StepDataObj(observation_dict=observation_dict,
                                                         action_int=action_int,
                                                         action_str=action_str,
                                                         reward_float=reward_float,
                                                         is_done=is_done,
                                                         info_dict=info_dict)

                reward_float: float = float(reward)
                rewards_list.append(reward_float)
                action_taken_list.append(action_int)

                # time.sleep(0.15)

                if is_done:
                    self._logger.info(f"Trial Number: {num_trial + 1}")
                    self._logger.info(f"Reached goal state in {episode_step_counter:,} steps")
                    self._logger.info("=" * 100)
                    observation_dict, info_dict = self._environment_obj.reset()

        self._environment_obj.close()

    def display_step_information(self, step_data_obj: StepDataObj) -> None:

        self._logger.info(
            f"action_int={step_data_obj.action_int}, "
            f"reward={step_data_obj.reward_float:.2f}, "
            f"action_str={step_data_obj.action_str}, "
            f"is_done={step_data_obj.is_done}, "
            f"info={step_data_obj.info_dict}"
        )

    def get_proximal_policy(self, observation_dict: dict) -> int:
        return 0
