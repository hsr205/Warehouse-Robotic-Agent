import time

import gymnasium as gym
from gymnasium import Env
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal
from minigrid.minigrid_env import MiniGridEnv
from typing_extensions import SupportsFloat

from logger.logger import AppLogger


class WareHouseEnv(MiniGridEnv):
    def __init__(
        self,
        size: int = 5,
        agent_start_position_tuple: tuple[int, int] = (1, 1),
        agent_start_direction: int = 0,
        max_steps: int | None = None,
        **kwargs,
    ) -> None:

        self._step_penalty: float = -0.01
        self._agent_start_direction = agent_start_direction
        self._agent_start_position_tuple = agent_start_position_tuple
        self._goal_position_tuple: tuple[int, int] | None = None

        mission_space: MissionSpace = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 50

        self._logger = AppLogger.get_logger(self.__class__.__name__)

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            max_steps=max_steps,
            see_through_walls=True,
            **kwargs,
        )

    def reset(self, *, seed=None, options=None):
        observation, info = super().reset(seed=seed, options=options)
        return observation, info

    @staticmethod
    def _gen_mission() -> str:
        return "Reach the goal"

    def _gen_grid(self, width: int, height: int) -> None:
        try:
            self._create_grid_world(width=width, height=height)

            self._goal_position_tuple = (width - 2, height - 2)
            self.put_obj(Goal(), *self._goal_position_tuple)

            self._place_agent_at_starting_position()

            self.mission = "Reach the goal"

        except Exception as e:
            self._logger.error(f"Exception thrown: {e}")

    def _create_grid_world(self, width: int, height: int) -> None:
        self.grid = Grid(width=width, height=height)
        self.grid.wall_rect(x=0, y=0, w=width, h=height)

    def _place_agent_at_starting_position(self) -> None:
        if self._agent_start_position_tuple is not None:
            self.agent_pos = self._agent_start_position_tuple
            self.agent_dir = self._agent_start_direction
        else:
            self.place_agent()

    def step(self, action_int: int) -> tuple:
        previous_distance_to_goal: int = self._get_manhattan_distance(
            position_tuple=self._goal_position_tuple
        )
        previous_agent_position_tuple: tuple[int, int] = self.agent_pos

        observation, reward, is_terminated, is_truncated, info = super().step(action_int)

        reward += self._step_penalty

        is_blocked_forward: bool = self._is_forward_collision(
            action=action_int,
            previous_agent_pos_tuple=previous_agent_position_tuple
        )

        if is_blocked_forward:
            reward -= 0.5
            info["collision"] = True
        else:
            info["collision"] = False

        reward = self._add_agent_incentive_towards_goal_state(
            reward=reward,
            previous_distance_to_goal=previous_distance_to_goal
        )

        if self._agent_reaches_goal_state():
            reward += 10.0
            is_terminated = True

        return observation, reward, is_terminated, is_truncated, info

    def _agent_reaches_goal_state(self) -> bool:
        return self.agent_pos == self._goal_position_tuple

    def _is_forward_collision(self, action, previous_agent_pos_tuple: tuple[int, int]) -> bool:
        is_forward_collision = (
            action == self.actions.forward and
            self.agent_pos == previous_agent_pos_tuple
        )
        return is_forward_collision

    def _add_agent_incentive_towards_goal_state(
        self,
        reward: SupportsFloat,
        previous_distance_to_goal: int,
    ) -> SupportsFloat:

        current_distance_to_goal: int = self._get_manhattan_distance(
            position_tuple=self._goal_position_tuple
        )

        if current_distance_to_goal < previous_distance_to_goal:
            reward += 0.30
        elif current_distance_to_goal > previous_distance_to_goal:
            reward -= 0.10
        else:
            reward -= 0.05

        return reward

    def _get_manhattan_distance(self, position_tuple: tuple[int, int]) -> int:
        current_x_coordinate, current_y_coordinate = self.agent_pos
        x_coordinate, y_coordinate = position_tuple

        return abs(current_x_coordinate - x_coordinate) + abs(current_y_coordinate - y_coordinate)

    def randomly_navigate_custom_grid_world(self) -> None:
        environment_obj: Env = WareHouseEnv(render_mode="human")
        observation_dict, info_dict = environment_obj.reset(seed=42)

        for _ in range(100):
            action_int = environment_obj.action_space.sample()
            observation_dict, reward_float, terminated_bool, truncated_bool, info_dict = environment_obj.step(
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
                observation_dict, info_dict = environment_obj.reset()

        environment_obj.close()

    def randomly_navigate_empty_grid_world(self) -> None:
        environment_obj: Env = gym.make(id="MiniGrid-Empty-5x5-v0", render_mode="human")
        observation_dict, info_dict = environment_obj.reset(seed=42)

        for _ in range(100):
            action_int = environment_obj.action_space.sample()
            observation_dict, reward_float, terminated_bool, truncated_bool, info_dict = environment_obj.step(
                action_int
            )

            time.sleep(0.1)

            if terminated_bool or truncated_bool:
                observation_dict, info_dict = environment_obj.reset()

        environment_obj.close()