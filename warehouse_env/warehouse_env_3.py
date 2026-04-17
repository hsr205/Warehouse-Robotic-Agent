import time
from typing import List, Tuple

from gymnasium import Env
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Wall, Ball, Box
from minigrid.minigrid_env import MiniGridEnv
from typing_extensions import SupportsFloat

from logger.logger import AppLogger


class WareHouseEnv3(MiniGridEnv):
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
        self._num_obstacles = num_obstacles
        self._agent_start_direction = agent_start_direction
        self._agent_start_position_tuple = agent_start_position_tuple

        self._is_carrying_package: bool = False
        self._package_position_list: list[tuple[int, int]] = []

        self._initial_obstacle_positions = obstacle_positions or []

        self._obstacles_list: List[dict] = []
        self._goal_position_tuple: Tuple[int, int] | None = None

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
        observation, info = super().reset(seed=seed, options=options)
        return observation, info

    @staticmethod
    def _gen_mission() -> str:
        return "Navigate the strange warehouse, pick up the package, and return to the goal"

    def _gen_grid(self, width: int, height: int) -> None:
        try:
            self._create_grid_world(width=width, height=height)

            self._build_weird_warehouse_layout(width=width, height=height)

            # Put the package deeper into the map and harder to access
            self._package_position_list = [(width - 5, height - 6)]

            for pickup_position_tuple in self._package_position_list:
                pickup_x_coordinate, pickup_y_coordinate = pickup_position_tuple
                self.pickup_object = Box(color="blue")
                self.grid.set(pickup_x_coordinate, pickup_y_coordinate, self.pickup_object)

            # Keep the goal far from the package
            self._goal_position_tuple = (width - 2, 1)
            self.put_obj(Goal(), *self._goal_position_tuple)

            # More obstacles, placed in awkward corridors
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
        """
        Builds a less uniform, more difficult warehouse layout.
        Uses a mix of vertical and horizontal walls with irregular openings.
        """

        # ---------------------------
        # Long vertical barriers
        # ---------------------------
        vertical_wall_specs: list[tuple[int, int, int, set[int]]] = [
            (3, 1, height - 2, {2, 9, 17}),
            (6, 2, height - 3, {5, 13, 19}),
            (9, 1, height - 4, {4, 10, 15}),
            # (13, 3, height - 2, {6, 14}),
            # (17, 1, height - 5, {8, 18}),
            # (20, 2, height - 2, {7, 16}),
        ]

        for x_coordinate, start_row, end_row, openings in vertical_wall_specs:
            for y_coordinate in range(start_row, end_row + 1):
                if y_coordinate not in openings:
                    self.grid.set(x_coordinate, y_coordinate, Wall())

        # ---------------------------
        # Strange isolated wall chunks
        # ---------------------------
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

            # NOTE - Remove for most working_demo_checkpoint_files_env_3/ files
            # (8, 22),
            (16, 22),
            # (18, 21),
            (19, 20),
            (22, 10),
            # (22, 4),
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

        # ---------------------------
        # Guard the package area with awkward walls
        # ---------------------------
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
        """
        Add moving obstacles.
        Some move horizontally, some vertically.
        """
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
        was_carrying_package_before_step: bool = self._is_carrying_package

        observation, reward, is_terminated, is_truncated, info = super().step(action_int)

        reward += self._step_penalty

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
            reward = 100
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

            # if not was_carrying_package_before_step:
            #     reward += 0.25
        else:
            reward = self._add_agent_incentive_to_move_toward_package(
                reward=reward,
                previous_agent_position_tuple=previous_agent_position_tuple,
            )

        info["collision"] = False

        return observation, reward, is_terminated, is_truncated, info

    def _move_obstacles(self) -> None:
        """
        Move each obstacle one step.
        Some move horizontally, some vertically.
        If blocked, reverse direction.
        """
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

    def _add_agent_incentive_to_move_toward_package(
            self,
            reward: SupportsFloat,
            previous_agent_position_tuple: tuple[int, int],
    ) -> SupportsFloat:

        if self._is_carrying_package is True:
            return reward

        if not self._package_position_list:
            return reward

        agent_x_coordinate, agent_y_coordinate = previous_agent_position_tuple

        for package_position_tuple in self._package_position_list:
            pickup_x_coordinate, pickup_y_coordinate = package_position_tuple

            current_distance_to_pickup: int = self._get_manhattan_distance(position_tuple=package_position_tuple)

            previous_distance_to_pickup = abs(agent_x_coordinate - pickup_x_coordinate) + abs(
                agent_y_coordinate - pickup_y_coordinate
            )

            if current_distance_to_pickup < previous_distance_to_pickup:
                reward += 15.5
            elif current_distance_to_pickup > previous_distance_to_pickup:
                reward -= 1.0

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
                reward += 35.0
                return reward

            if not is_action_pickup and is_agent_in_valid_pickup_location:
                reward -= 0.25
                return reward

            if is_action_pickup and not is_agent_in_valid_pickup_location:
                reward -= 0.2
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

        if current_distance_to_goal < previous_distance_to_goal:
            reward += 22.5
        elif current_distance_to_goal > previous_distance_to_goal:
            reward -= 4.0

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

    def _get_manhattan_distance(self, position_tuple: tuple[int, int]) -> int:
        current_x_coordinate, current_y_coordinate = self.agent_pos
        x_coordinate, y_coordinate = position_tuple

        return abs(current_x_coordinate - x_coordinate) + abs(current_y_coordinate - y_coordinate)

    def randomly_navigate_custom_grid_world(self) -> None:
        environment_obj: Env = WareHouseEnv3(render_mode="human")
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
