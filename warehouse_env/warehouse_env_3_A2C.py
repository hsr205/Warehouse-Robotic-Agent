import time
from typing import List, Tuple

import numpy as np
from gymnasium import Env
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Wall, Ball, Box
from minigrid.minigrid_env import MiniGridEnv
from typing_extensions import SupportsFloat

from logger.logger import AppLogger


class WareHouseEnv3A2C(MiniGridEnv):
    def __init__(
            self,
            size: int = 24,
            agent_start_position_tuple: tuple[int, int] = (1, 1),
            agent_start_direction: int = 0,
            max_steps: int | None = None,
            num_obstacles: int = 12,
            obstacle_positions: List[Tuple[int, int]] | None = None,
            **kwargs,
    ) -> None:

        self._step_penalty: float = -0.01
        self._distance_shaping_scale: float = 3.0
        self._num_obstacles = num_obstacles
        self._agent_start_direction = agent_start_direction
        self._agent_start_position_tuple = agent_start_position_tuple

        self._is_carrying_package: bool = False
        self._package_position_list: list[tuple[int, int]] = []
        self._stationary_turn_count: int = 0

        self._initial_obstacle_positions = obstacle_positions or []

        self._obstacles_list: List[dict] = []
        self._goal_position_tuple: Tuple[int, int] | None = None
        self._last_step_info_dict: dict = {}

        mission_space: MissionSpace = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 6 * (size ** 2)

        self._logger = AppLogger.get_logger(self.__class__.__name__)

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            max_steps=max_steps,
            see_through_walls=True,
            **kwargs,
        )

    def reset(self, *, seed=None, options=None):
        self._is_carrying_package = False
        self._stationary_turn_count = 0
        self._last_step_info_dict = {}
        observation, info = super().reset(seed=seed, options=options)
        return observation, info

    @staticmethod
    def _gen_mission() -> str:
        return "Navigate the strange warehouse, pick up the package, and return to the goal"

    def _gen_grid(self, width: int, height: int) -> None:
        try:
            self._create_grid_world(width=width, height=height)

            self._build_weird_warehouse_layout(width=width, height=height)

            self._package_position_list = [(width - 5, height - 6)]

            for pickup_position_tuple in self._package_position_list:
                pickup_x_coordinate, pickup_y_coordinate = pickup_position_tuple
                self.pickup_object = Box(color="blue")
                self.grid.set(pickup_x_coordinate, pickup_y_coordinate, self.pickup_object)

            self._goal_position_tuple = (width - 2, 1)
            self.put_obj(Goal(), *self._goal_position_tuple)

            self._initial_obstacle_positions = [
                (5, 7),
                (8, 4),
                (10, 13),
                (14, 17),
                (16, 8),
                (19, 5),
            ]
            self._num_obstacles = len(self._initial_obstacle_positions)
            self._place_dynamic_obstacles()

            self._place_agent_at_starting_position()

            self.mission = "Navigate the strange warehouse, pick up the package, and reach the goal with the package"

        except Exception as e:
            self._logger.error(f"Exception thrown in _gen_grid: {e}")

    def _create_grid_world(self, width: int, height: int) -> None:
        self.grid = Grid(width=width, height=height)
        self.grid.wall_rect(x=0, y=0, w=width, h=height)

    def _build_weird_warehouse_layout(self, width: int, height: int) -> None:
        vertical_wall_specs: list[tuple[int, int, int, set[int]]] = [
            (3, 1, height - 2, {2, 9, 17}),
            (6, 2, height - 3, {5, 13, 19}),
            (9, 1, height - 4, {4, 10, 15}),
        ]

        for x_coordinate, start_row, end_row, openings in vertical_wall_specs:
            for y_coordinate in range(start_row, end_row + 1):
                if y_coordinate not in openings:
                    self.grid.set(x_coordinate, y_coordinate, Wall())

        isolated_wall_positions: list[tuple[int, int]] = [
            (16, 2), (17, 3), (18, 2),
            (13, 5), (14, 4),
            (13, 8), (14, 7),
            (13, 10),
            (14, 11),
            (13, 12),
            (14, 13),
            (13, 14),
            (14, 15),
            (18, 9),
            (19, 10),
            (20, 9),
            (18, 11),
            (20, 11),
            (17, 19),
            (11, 17), (12, 18),
            (21, 20), (21, 21),
            (12, 20), (13, 21), (14, 20),
            (16, 22),
            (19, 20),
            (22, 10),
            (17, 7),
            (18, 6),
            (19, 7),
            (20, 6),
            (21, 14),
            (21, 7),
        ]

        for x_coordinate, y_coordinate in isolated_wall_positions:
            if self.grid.get(x_coordinate, y_coordinate) is None:
                self.grid.set(x_coordinate, y_coordinate, Wall())

        package_zone_walls: list[tuple[int, int]] = [
            (width - 7, height - 8),
            (width - 6, height - 8),
            (width - 5, height - 8),
            (width - 4, height - 8),
            (width - 7, height - 7),
            (width - 7, height - 6)
        ]

        for x_coordinate, y_coordinate in package_zone_walls:
            if self.grid.get(x_coordinate, y_coordinate) is None:
                self.grid.set(x_coordinate, y_coordinate, Wall())

    def _place_dynamic_obstacles(self) -> None:
        self._obstacles_list = []

        for idx in range(self._num_obstacles):
            if idx < len(self._initial_obstacle_positions):
                x_coordinate, y_coordinate = self._initial_obstacle_positions[idx]
            else:
                x_coordinate = 2 + idx
                y_coordinate = 2 + idx

            if self.grid.get(x_coordinate, y_coordinate) is not None:
                continue

            if self._goal_position_tuple is not None and (x_coordinate, y_coordinate) == self._goal_position_tuple:
                continue

            if (x_coordinate, y_coordinate) == self._agent_start_position_tuple:
                continue

            obstacle_obj = Ball(color="red")
            self.grid.set(x_coordinate, y_coordinate, obstacle_obj)

            is_vertical_movement: bool = idx % 2 == 0

            self._obstacles_list.append(
                {
                    "pos": (x_coordinate, y_coordinate),
                    "dir": 1,
                    "axis": "vertical" if is_vertical_movement else "horizontal",
                    "obj": obstacle_obj,
                }
            )

    def _place_agent_at_starting_position(self) -> None:
        if self._agent_start_position_tuple is not None:
            self.agent_pos = self._agent_start_position_tuple
            self.agent_dir = self._agent_start_direction
        else:
            self.place_agent()

    def step(self, action_int: int) -> tuple:
        previous_distance_to_goal: int = self._get_manhattan_distance(position_tuple=self._goal_position_tuple)
        previous_agent_position_tuple: tuple[int, int] = self.agent_pos
        previous_agent_direction_int: int = self.agent_dir
        was_carrying_package_before_step: bool = self._is_carrying_package

        observation, reward, is_terminated, is_truncated, info = super().step(action_int)
        info["valid_pickup_location"] = self._is_agent_in_valid_pickup_location()
        info["missed_pickup_opportunity"] = False
        info["invalid_pickup_attempt"] = False
        info["forward_collision"] = False
        self._last_step_info_dict = info

        reward += self._step_penalty

        if self._is_forward_collision(action=action_int, previous_agent_pos_tuple=previous_agent_position_tuple):
            reward -= 1.0
            info["forward_collision"] = True

        if self._agent_hits_obstacle():
            reward = -2.0
            is_terminated = True
            info["collision"] = True
            self._is_carrying_package = False
            return observation, reward, is_terminated, is_truncated, info

        if is_terminated or is_truncated:
            return observation, reward, is_terminated, is_truncated, info

        self._move_obstacles()
        observation = self.gen_obs()

        if self._agent_hits_obstacle():
            reward = -2.0
            is_terminated = True
            info["collision"] = True
            self._is_carrying_package = False
            return observation, reward, is_terminated, is_truncated, info

        is_agent_allowed_in_goal_state: bool = self.agent_pos == self._goal_position_tuple and not self._is_carrying_package

        if is_agent_allowed_in_goal_state:
            self.agent_pos = previous_agent_position_tuple
            reward = -0.50
            is_terminated = False
            is_truncated = False
            observation = self.gen_obs()
            info["blocked_goal_without_package"] = True
        else:
            info["blocked_goal_without_package"] = False

        reward = self._agent_incentive_to_pickup_package(reward=reward, action_int=action_int)

        if self._agent_reaches_goal_state() and self._is_carrying_package:
            reward = 50
            is_terminated = True
            info["collision"] = False
            self._is_carrying_package = False
            return observation, reward, is_terminated, is_truncated, info

        if self._is_carrying_package:
            reward = self._add_agent_incentive_towards_goal_state(
                reward=reward,
                previous_distance_to_goal=previous_distance_to_goal,
                previous_agent_position_tuple=previous_agent_position_tuple,
            )

            if not was_carrying_package_before_step:
                reward += 0.25
        else:
            reward = self._add_agent_incentive_to_move_toward_package(
                reward=reward,
                previous_agent_position_tuple=previous_agent_position_tuple,
                previous_agent_direction_int=previous_agent_direction_int,
            )

        reward = self._add_stationary_turn_penalty(
            reward=reward,
            action_int=action_int,
            previous_agent_position_tuple=previous_agent_position_tuple,
        )

        info["collision"] = False

        return observation, reward, is_terminated, is_truncated, info

    def _move_obstacles(self) -> None:
        for obstacle in self._obstacles_list:
            old_x_coordinate, old_y_coordinate = obstacle["pos"]
            direction = obstacle["dir"]
            axis = obstacle["axis"]

            if axis == "horizontal":
                candidate_x_coordinate = old_x_coordinate + direction
                candidate_y_coordinate = old_y_coordinate
            else:
                candidate_x_coordinate = old_x_coordinate
                candidate_y_coordinate = old_y_coordinate + direction

            can_move: bool = self._is_valid_obstacle_cell(candidate_x_coordinate, candidate_y_coordinate)

            if not can_move:
                direction *= -1

                if axis == "horizontal":
                    candidate_x_coordinate = old_x_coordinate + direction
                    candidate_y_coordinate = old_y_coordinate
                else:
                    candidate_x_coordinate = old_x_coordinate
                    candidate_y_coordinate = old_y_coordinate + direction

                if not self._is_valid_obstacle_cell(candidate_x_coordinate, candidate_y_coordinate):
                    obstacle["dir"] = direction
                    continue

            if self._goal_position_tuple == (old_x_coordinate, old_y_coordinate):
                self.grid.set(old_x_coordinate, old_y_coordinate, Goal())
            else:
                self.grid.set(old_x_coordinate, old_y_coordinate, None)

            self.grid.set(candidate_x_coordinate, candidate_y_coordinate, obstacle["obj"])
            obstacle["pos"] = (candidate_x_coordinate, candidate_y_coordinate)
            obstacle["dir"] = direction

    def _is_valid_obstacle_cell(self, x: int, y: int) -> bool:
        if x <= 0 or y <= 0 or x >= self.width - 1 or y >= self.height - 1:
            return False

        if (x, y) == self.agent_pos:
            return False

        if self._goal_position_tuple is not None and (x, y) == self._goal_position_tuple:
            return False

        cell_obj = self.grid.get(x, y)
        if cell_obj is not None:
            return False

        return True

    def _agent_hits_obstacle(self) -> bool:
        return any(obstacle["pos"] == self.agent_pos for obstacle in self._obstacles_list)

    def _agent_reaches_goal_state(self) -> bool:
        return self.agent_pos == self._goal_position_tuple

    def _is_forward_collision(self, action, previous_agent_pos_tuple: tuple[int, int]) -> bool:
        is_forward_collision = (
                action == self.actions.forward and
                self.agent_pos == previous_agent_pos_tuple
        )
        return is_forward_collision

    def get_action_mask(self) -> np.ndarray:
        is_pickup_available = (not self._is_carrying_package) and self._is_agent_in_valid_pickup_location()
        return np.array([1, 1, 1, int(is_pickup_available)], dtype=np.float32)

    def _add_agent_incentive_to_move_toward_package(
            self,
            reward: SupportsFloat,
            previous_agent_position_tuple: tuple[int, int],
            previous_agent_direction_int: int,
    ) -> SupportsFloat:

        if self._is_carrying_package is True:
            return reward

        if not self._package_position_list:
            return reward

        previous_pickup_configuration_distance: float = self._get_pickup_configuration_distance(
            agent_position_tuple=previous_agent_position_tuple,
            agent_direction_int=previous_agent_direction_int
        )
        current_pickup_configuration_distance: float = self._get_pickup_configuration_distance(
            agent_position_tuple=self.agent_pos,
            agent_direction_int=self.agent_dir
        )

        distance_delta_to_pickup_configuration = (
            previous_pickup_configuration_distance - current_pickup_configuration_distance
        )
        reward += self._distance_shaping_scale * distance_delta_to_pickup_configuration

        return reward

    def _agent_incentive_to_pickup_package(self, reward: SupportsFloat, action_int: int) -> SupportsFloat:
        if not self._is_carrying_package:
            is_action_pickup: bool = action_int == 3
            is_agent_in_valid_pickup_location: bool = self._is_agent_in_valid_pickup_location()

            if is_action_pickup and is_agent_in_valid_pickup_location:
                self._is_carrying_package = True

                package_x_coordinate: int = self._package_position_list[0][0]
                package_y_coordinate: int = self._package_position_list[0][1]

                self.grid.set(i=package_x_coordinate, j=package_y_coordinate, v=None)
                reward += 60
                return reward

            if not is_action_pickup and is_agent_in_valid_pickup_location:
                reward -= 5.0
                self._last_step_info_dict["missed_pickup_opportunity"] = True
                return reward

            if is_action_pickup and not is_agent_in_valid_pickup_location:
                reward -= 1.0
                self._last_step_info_dict["invalid_pickup_attempt"] = True
                return reward

        if self._is_carrying_package and action_int == 3:
            reward -= 1.0
            self._last_step_info_dict["invalid_pickup_attempt"] = True
            return reward

        return reward

    def _add_agent_incentive_towards_goal_state(
            self,
            reward: SupportsFloat,
            previous_distance_to_goal: int,
            previous_agent_position_tuple: tuple[int, int],
    ) -> SupportsFloat:

        if self.agent_pos == previous_agent_position_tuple:
            return reward

        current_distance_to_goal: int = self._get_manhattan_distance(position_tuple=self._goal_position_tuple)
        distance_delta_to_goal = previous_distance_to_goal - current_distance_to_goal
        reward += self._distance_shaping_scale * distance_delta_to_goal

        return reward

    def _add_stationary_turn_penalty(
            self,
            reward: SupportsFloat,
            action_int: int,
            previous_agent_position_tuple: tuple[int, int],
    ) -> SupportsFloat:
        if not self._is_carrying_package:
            self._stationary_turn_count = 0
            return reward

        is_turn_action: bool = action_int in (self.actions.left, self.actions.right)
        is_stationary: bool = self.agent_pos == previous_agent_position_tuple

        if is_turn_action and is_stationary:
            self._stationary_turn_count += 1
        else:
            self._stationary_turn_count = 0
            return reward

        if self._stationary_turn_count >= 3:
            reward -= 1.0

        return reward

    def _is_agent_in_valid_pickup_location(self) -> bool:
        agent_x_coordinate: int = self.agent_pos[0]
        agent_y_coordinate: int = self.agent_pos[1]

        package_x_coordinate: int = self._package_position_list[0][0]
        package_y_coordinate: int = self._package_position_list[0][1]

        is_agent_direction_right: bool = self.agent_dir == 0
        is_agent_direction_down: bool = self.agent_dir == 1
        is_agent_direction_left: bool = self.agent_dir == 2
        is_agent_direction_up: bool = self.agent_dir == 3

        is_agent_and_package_on_same_row: bool = agent_y_coordinate == package_y_coordinate
        is_agent_and_package_on_same_column: bool = agent_x_coordinate == package_x_coordinate

        if is_agent_direction_down and is_agent_and_package_on_same_column and agent_y_coordinate + 1 == package_y_coordinate:
            return True

        if is_agent_direction_right and is_agent_and_package_on_same_row and agent_x_coordinate + 1 == package_x_coordinate:
            return True

        if is_agent_direction_left and is_agent_and_package_on_same_row and agent_x_coordinate - 1 == package_x_coordinate:
            return True

        if is_agent_direction_up and is_agent_and_package_on_same_column and agent_y_coordinate - 1 == package_y_coordinate:
            return True

        return False

    @staticmethod
    def _get_direction_distance(source_direction_int: int, target_direction_int: int) -> int:
        clockwise_turns_int = (target_direction_int - source_direction_int) % 4
        counterclockwise_turns_int = (source_direction_int - target_direction_int) % 4
        return min(clockwise_turns_int, counterclockwise_turns_int)

    def _get_pickup_configuration_distance(
            self,
            agent_position_tuple: tuple[int, int],
            agent_direction_int: int,
    ) -> float:
        valid_pickup_states_list: list[tuple[tuple[int, int], int]] = []

        for package_x_coordinate, package_y_coordinate in self._package_position_list:
            valid_pickup_states_list.extend([
                ((package_x_coordinate - 1, package_y_coordinate), 0),
                ((package_x_coordinate + 1, package_y_coordinate), 2),
                ((package_x_coordinate, package_y_coordinate - 1), 1),
                ((package_x_coordinate, package_y_coordinate + 1), 3),
            ])

        distance_to_best_pickup_state_float: float = float("inf")

        for pickup_position_tuple, required_direction_int in valid_pickup_states_list:
            pickup_x_coordinate, pickup_y_coordinate = pickup_position_tuple
            agent_x_coordinate, agent_y_coordinate = agent_position_tuple

            position_distance_int = abs(agent_x_coordinate - pickup_x_coordinate) + abs(
                agent_y_coordinate - pickup_y_coordinate
            )
            direction_distance_int = self._get_direction_distance(
                source_direction_int=agent_direction_int,
                target_direction_int=required_direction_int
            )

            total_distance_float = position_distance_int + 0.5 * direction_distance_int
            distance_to_best_pickup_state_float = min(distance_to_best_pickup_state_float, total_distance_float)

        return distance_to_best_pickup_state_float

    def _get_manhattan_distance(self, position_tuple: tuple[int, int]) -> int:
        current_x_coordinate, current_y_coordinate = self.agent_pos
        x_coordinate, y_coordinate = position_tuple

        return abs(current_x_coordinate - x_coordinate) + abs(current_y_coordinate - y_coordinate)

    def randomly_navigate_custom_grid_world(self) -> None:
        environment_obj: Env = WareHouseEnv3A2C(render_mode="human")
        observation_dict, info_dict = environment_obj.reset(seed=42)

        for _ in range(300):
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
