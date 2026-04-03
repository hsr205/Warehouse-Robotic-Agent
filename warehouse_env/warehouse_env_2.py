import time
from typing import List, Tuple

import gymnasium as gym
from gymnasium import Env
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Wall, Ball, Box
from minigrid.minigrid_env import MiniGridEnv
from typing_extensions import SupportsFloat

from logger.logger import AppLogger


class WareHouseEnv2(MiniGridEnv):
    def __init__(
            self,
            size: int = 20,
            agent_start_position_tuple: tuple[int, int] = (1, 1),
            agent_start_direction: int = 0,
            max_steps: int | None = None,
            num_obstacles: int = 6,
            obstacle_positions: List[Tuple[int, int]] | None = None,
            **kwargs,
    ) -> None:

        self._step_penalty: float = -0.01
        self._num_obstacles = num_obstacles
        self._agent_start_direction = agent_start_direction
        self._agent_start_position_tuple = agent_start_position_tuple

        self._is_carrying_package: bool = False
        self._package_position_list: list[tuple[int, int]] = []

        self._initial_obstacle_positions = obstacle_positions or [(2, 6), (10, 10)]

        self._obstacles_list: List[dict] = []
        self._goal_position_tuple: Tuple[int, int] | None = None

        mission_space: MissionSpace = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * (size ** 2)

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
        observation, info = super().reset(seed=seed, options=options)
        return observation, info

    @staticmethod
    def _gen_mission() -> str:
        return "Navigate the Warehouse -> Reach The Goal"

    def _gen_grid(self, width: int, height: int) -> None:
        try:
            self._create_grid_world(width=width, height=height)

            self._package_position_list = [
                (11, 12)
            ]

            for pickup_position_tuple in self._package_position_list:
                pickup_x_coordinate, pickup_y_coordinate = pickup_position_tuple
                self.pickup_object = Box(color="blue")
                self.grid.set(pickup_x_coordinate, pickup_y_coordinate, self.pickup_object)

            self._goal_position_tuple = (width - 2, height - 2)
            self.put_obj(Goal(), *self._goal_position_tuple)

            self._initial_obstacle_positions = [
                (2, 4),
                (4, 10),
                (7, 3),
                (12, 14),
                (13, 6),
                (16, 16),
            ]
            self._num_obstacles = len(self._initial_obstacle_positions)
            self._place_dynamic_obstacles()

            shelf_aisle_columns_list: list[int] = [3, 6, 9, 12, 15]
            agent_crossing_rows_list: list[int] = [4, 8, 12, 16]

            for column_num in shelf_aisle_columns_list:
                for row_num in range(1, height - 1):
                    if row_num not in agent_crossing_rows_list:
                        self.grid.set(i=column_num, j=row_num, v=Wall())

            self._place_agent_at_starting_position()

            self.mission = "Navigate Warehouse Grid 2, Pickup Package, Reach Goal with Package"

        except Exception as e:
            self._logger.error(f"Exception thrown in _gen_grid_2: {e}")

    def _create_grid_world(self, width: int, height: int) -> None:
        self.grid = Grid(width=width, height=height)
        self.grid.wall_rect(x=0, y=0, w=width, h=height)

    def _place_dynamic_obstacles(self) -> None:
        """
        Add moving obstacles to the warehouse.
        Each obstacle moves horizontally and bounces when blocked.
        """
        self._obstacles_list: list[dict] = []

        for idx in range(self._num_obstacles):
            if idx < len(self._initial_obstacle_positions):
                pos = self._initial_obstacle_positions[idx]
            else:
                pos = (2 + idx, 2 + idx)

            x_coordinate, y_coordinate = pos

            # Safety check so obstacles do not start inside walls or on the goal
            if self.grid.get(x_coordinate, y_coordinate) is not None:
                continue
            if self._goal_position_tuple is not None and (x_coordinate, y_coordinate) == self._goal_position_tuple:
                continue
            if (x_coordinate, y_coordinate) == self._agent_start_position_tuple:
                continue

            obstacle_obj: Ball = Ball(color="red")
            self.grid.set(x_coordinate, y_coordinate, obstacle_obj)

            self._obstacles_list.append(
                {
                    "pos": (x_coordinate, y_coordinate),
                    "dir": 1,  # +1 means move right, -1 means move left
                    "obj": obstacle_obj,
                }
            )

    def _place_agent_at_starting_position(self) -> None:
        if self._agent_start_position_tuple is not None:
            self.agent_pos: tuple[int, int] = self._agent_start_position_tuple
            self.agent_dir: int = self._agent_start_direction
        else:
            self.place_agent()

    def step(self, action_int: int) -> tuple:
        """
        One environment step:
        1. Move agent using MiniGrid logic
        2. Check collision with obstacle
        3. Move obstacles
        4. Check collision again
        5. Apply custom reward shaping
        """

        previous_distance_to_goal: int = self._get_manhattan_distance(position_tuple=self._goal_position_tuple)

        previous_agent_position_tuple: tuple[int, int] = self.agent_pos

        was_carrying_package_before_step: bool = self._is_carrying_package

        observation, reward, is_terminated, is_truncated, info = super().step(action_int)

        # Small step penalty to encourage efficiency
        reward += self._step_penalty

        # Case 1: agent moves into obstacle
        if self._agent_hits_obstacle():
            reward = -2.0
            is_terminated = True
            info["collision"] = True
            self._is_carrying_package = False
            return observation, reward, is_terminated, is_truncated, info

        # If goal already reached, do not move obstacles
        if is_terminated or is_truncated:
            return observation, reward, is_terminated, is_truncated, info

        # Move obstacles after the agent acts
        self._move_obstacles()

        observation = self.gen_obs()

        # Case 2: obstacle moves into agent
        if self._agent_hits_obstacle():
            reward = -2.0
            is_terminated = True
            info["collision"] = True
            self._is_carrying_package = False
            return observation, reward, is_terminated, is_truncated, info

        # Case 3: If the agent moves into the goal state and is not carrying a package
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
        # Case 4: agent reaches goal state
        if self._agent_reaches_goal_state() and self._is_carrying_package:
            reward = 30
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
            )

        info["collision"] = False

        return observation, reward, is_terminated, is_truncated, info

    def _move_obstacles(self) -> None:
        """
        Move each obstacle one step horizontally.
        If blocked by a wall or object, reverse direction.
        """
        for obstacle in self._obstacles_list:
            old_x, old_y = obstacle["pos"]
            direction = obstacle["dir"]

            candidate_x = old_x + direction
            candidate_y = old_y

            can_move = self._is_valid_obstacle_cell(candidate_x, candidate_y)

            if not can_move:
                direction *= -1
                candidate_x = old_x + direction
                candidate_y = old_y

                if not self._is_valid_obstacle_cell(candidate_x, candidate_y):
                    obstacle["dir"] = direction
                    continue

            if self._goal_position_tuple == (old_x, old_y):
                self.grid.set(old_x, old_y, Goal())
            else:
                self.grid.set(old_x, old_y, None)

            self.grid.set(candidate_x, candidate_y, obstacle["obj"])
            obstacle["pos"] = (candidate_x, candidate_y)
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

    def _add_agent_incentive_to_move_toward_package(self, reward: SupportsFloat,
                                                    previous_agent_position_tuple: tuple[int, int]) -> SupportsFloat:

        if self._is_carrying_package is True:
            return reward

        if not self._package_position_list:
            return reward

        agent_x_coordinate, agent_y_coordinate = previous_agent_position_tuple

        for package_position_tuple in self._package_position_list:

            pickup_x_coordinate, pickup_y_coordinate = package_position_tuple

            current_distance_to_pickup: int = self._get_manhattan_distance(position_tuple=package_position_tuple)

            previous_distance_to_pickup = abs(agent_x_coordinate - pickup_x_coordinate) + abs(
                agent_y_coordinate - pickup_y_coordinate)

            if current_distance_to_pickup < previous_distance_to_pickup:
                reward += 1.0
            elif current_distance_to_pickup > previous_distance_to_pickup:
                reward -= 0.2

        return reward

    def _agent_incentive_to_pickup_package(self, reward: SupportsFloat, action_int: int) -> SupportsFloat:

        if not self._is_carrying_package:

            # TODO: In the event of multiple packages this needs to change -> self._package_position_list[0]

            is_action_pickup: bool = action_int == 3

            is_agent_in_valid_pickup_location: bool = self._is_agent_in_valid_pickup_location()

            if is_action_pickup and is_agent_in_valid_pickup_location:
                self._is_carrying_package = True
                package_x_coordinate: int = self._package_position_list[0][0]
                package_y_coordinate: int = self._package_position_list[0][1]

                self.grid.set(i=package_x_coordinate, j=package_y_coordinate, v=None)

                reward += 22.5

                return reward

            if not is_action_pickup and is_agent_in_valid_pickup_location:
                reward -= 0.25

                return reward

            if is_action_pickup and not is_agent_in_valid_pickup_location:
                reward -= 0.2

                return reward

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

    def _add_agent_incentive_towards_goal_state(self, reward: SupportsFloat,
                                                previous_distance_to_goal: int,
                                                previous_agent_position_tuple: tuple[int, int]) -> SupportsFloat:

        if self.agent_pos == previous_agent_position_tuple:
            return reward

        current_distance_to_goal: int = self._get_manhattan_distance(position_tuple=self._goal_position_tuple)

        if current_distance_to_goal < previous_distance_to_goal:
            reward += 7.5
        elif current_distance_to_goal > previous_distance_to_goal:
            reward -= 2.5

        return reward

    def _get_manhattan_distance(self, position_tuple: tuple[int, int]) -> int:
        current_x_coordinate, current_y_coordinate = self.agent_pos
        x_coordinate, y_coordinate = position_tuple

        return abs(current_x_coordinate - x_coordinate) + abs(current_y_coordinate - y_coordinate)

    def randomly_navigate_custom_grid_world(self) -> None:
        environment_obj: Env = WareHouseEnv2(render_mode="human")
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
