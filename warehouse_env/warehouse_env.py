import time
from typing import List, Tuple

import gymnasium as gym
from gymnasium import Env
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Wall, Ball
from minigrid.minigrid_env import MiniGridEnv

from logger.logger import AppLogger


class WareHouseEnv(MiniGridEnv):
    def __init__(
            self,
            size: int = 16,
            agent_start_position_tuple: tuple[int, int] = (1, 1),
            agent_start_direction: int = 0,
            max_steps: int | None = None,
            num_obstacles: int = 2,
            obstacle_positions: List[Tuple[int, int]] | None = None,
            **kwargs,
    ) -> None:

        self.step_penalty: float = -0.01
        self.num_obstacles = num_obstacles
        self.agent_start_direction = agent_start_direction
        self.agent_start_position_tuple = agent_start_position_tuple

        # Fixed obstacle start positions if not provided
        self.initial_obstacle_positions = obstacle_positions or [(2, 6), (10, 10)]

        # Will be reset each episode
        self.obstacles_list: List[dict] = []
        self.goal_position_tuple: Tuple[int, int] | None = None

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

    def _gen_grid(self, width: int, height: int) -> None:
        self._create_grid_world(width=width, height=height)
        self._add_grid_elements(width=width, height=height)

        # Goal position
        self.goal_position_tuple = (width - 2, height - 2)
        self.put_obj(Goal(), *self.goal_position_tuple)

        # Place dynamic obstacles
        self._place_dynamic_obstacles()

        # Place agent
        self._place_agent_at_starting_position()

        self.mission = "Navigate Warehouse, Avoid Obstacles, Reach Goal"

    def _create_grid_world(self, width: int, height: int) -> None:
        self.grid = Grid(width=width, height=height)
        self.grid.wall_rect(x=0, y=0, w=width, h=height)

    def _add_grid_elements(self, width: int, height: int) -> None:
        """
        Create a warehouse-style layout with vertical shelf aisles.
        Gaps allow the agent and obstacles to move between aisles.
        """
        shelf_aisle_columns_list: list[int] = [3, 6, 9, 12]
        agent_crossing_rows_list: list[int] = [4, 8, 12]

        for column_num in shelf_aisle_columns_list:
            for row_num in range(1, height - 1):
                if row_num not in agent_crossing_rows_list:
                    self.grid.set(i=column_num, j=row_num, v=Wall())

    def _place_dynamic_obstacles(self) -> None:
        """
        Add moving obstacles to the warehouse.
        Each obstacle moves horizontally and bounces when blocked.
        """
        self.obstacles_list: list[dict] = []

        for idx in range(self.num_obstacles):
            if idx < len(self.initial_obstacle_positions):
                pos = self.initial_obstacle_positions[idx]
            else:
                pos = (2 + idx, 2 + idx)

            x_coordinate, y_coordinate = pos

            # Safety check so obstacles do not start inside walls or on the goal
            if self.grid.get(x_coordinate, y_coordinate) is not None:
                continue
            if self.goal_position_tuple is not None and (x_coordinate, y_coordinate) == self.goal_position_tuple:
                continue
            if (x_coordinate, y_coordinate) == self.agent_start_position_tuple:
                continue

            obstacle_obj: Ball = Ball(color="red")
            self.grid.set(x_coordinate, y_coordinate, obstacle_obj)

            self.obstacles_list.append(
                {
                    "pos": (x_coordinate, y_coordinate),
                    "dir": 1,  # +1 means move right, -1 means move left
                    "obj": obstacle_obj,
                }
            )

    def _place_agent_at_starting_position(self) -> None:
        if self.agent_start_position_tuple is not None:
            self.agent_pos = self.agent_start_position_tuple
            self.agent_dir = self.agent_start_direction
        else:
            self.place_agent()

    @staticmethod
    def _gen_mission() -> str:
        return "Navigate the Warehouse -> Reach The Goal"

    def _move_obstacles(self) -> None:
        """
        Move each obstacle one step horizontally.
        If blocked by a wall or object, reverse direction.
        """
        for obstacle in self.obstacles_list:
            old_x, old_y = obstacle["pos"]
            direction = obstacle["dir"]

            candidate_x = old_x + direction
            candidate_y = old_y

            can_move = self._is_valid_obstacle_cell(candidate_x, candidate_y)

            if not can_move:
                # Reverse direction and try once more
                direction *= -1
                candidate_x = old_x + direction
                candidate_y = old_y

                if not self._is_valid_obstacle_cell(candidate_x, candidate_y):
                    obstacle["dir"] = direction
                    continue

            # Clear old obstacle cell
            if self.goal_position_tuple == (old_x, old_y):
                self.grid.set(old_x, old_y, Goal())
            else:
                self.grid.set(old_x, old_y, None)

            # Move obstacle to new cell
            self.grid.set(candidate_x, candidate_y, obstacle["obj"])
            obstacle["pos"] = (candidate_x, candidate_y)
            obstacle["dir"] = direction

    def _is_valid_obstacle_cell(self, x: int, y: int) -> bool:
        """
        Obstacles may move into empty cells only.
        They should not move into walls, the goal, or other occupied cells.
        They also should not move onto the agent.
        """
        if x <= 0 or y <= 0 or x >= self.width - 1 or y >= self.height - 1:
            return False

        if (x, y) == self.agent_pos:
            return False

        if self.goal_position_tuple is not None and (x, y) == self.goal_position_tuple:
            return False

        cell_obj = self.grid.get(x, y)
        if cell_obj is not None:
            return False

        return True

    def _agent_hits_obstacle(self) -> bool:
        return any(obstacle["pos"] == self.agent_pos for obstacle in self.obstacles_list)

    def _agent_reaches_goal_state(self) -> bool:
        return self.agent_pos == self.goal_position_tuple

    def step(self, action):
        """
        One environment step:
        1. Move agent using MiniGrid logic
        2. Check collision with obstacle
        3. Move obstacles
        4. Check collision again
        5. Apply custom reward shaping
        """
        observation, reward, is_terminated, is_truncated, info = super().step(action)

        # Small step penalty to encourage efficiency
        reward += self.step_penalty

        # Case 1: agent moves into obstacle
        if self._agent_hits_obstacle():
            reward = -1.0
            is_terminated = False
            info["collision"] = True
            return observation, reward, is_terminated, is_truncated, info

        # If goal already reached, do not move obstacles
        if is_terminated or is_truncated:
            return observation, reward, is_terminated, is_truncated, info

        # Move obstacles after the agent acts
        self._move_obstacles()

        # Refresh observation after obstacle movement
        observation = self.gen_obs()

        # Case 2: obstacle moves into agent
        if self._agent_hits_obstacle():
            reward = -1.0
            is_terminated = False
            info["collision"] = True
            return observation, reward, is_terminated, is_truncated, info

        # Case 3: agent reaches goal state
        # TODO: After testing remove this, the goal is to deliver packages to
        #       the correct locations while navigating through the environment optimally
        if self._agent_reaches_goal_state():
            reward = 1.0
            is_terminated = True
            info["collision"] = False
            return observation, reward, is_terminated, is_truncated, info

        # TODO: Create more cases such as
        #  (1) If the agent picks up an item
        #  (2) If the agent drops off the item at the correct location
        #  (3) If the agent drops the item (need to check if this is possible)
        #  (4) If the agent drops the item in the wrong location

        info["collision"] = False

        return observation, reward, is_terminated, is_truncated, info

    def randomly_navigate_custom_grid_world(self) -> None:
        environment_obj: Env = WareHouseEnv(render_mode="human")
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

    def randomly_navigate_empty_grid_world(self) -> None:
        environment_obj: Env = gym.make(id="MiniGrid-Empty-16x16-v0", render_mode="human")
        observation_dict, info_dict = environment_obj.reset(seed=42)

        for _ in range(200):
            action_int = environment_obj.action_space.sample()
            observation_dict, reward_float, terminated_bool, truncated_bool, info_dict = environment_obj.step(
                action_int
            )

            time.sleep(0.1)

            if terminated_bool or truncated_bool:
                observation_dict, info_dict = environment_obj.reset()

        environment_obj.close()
