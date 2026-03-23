import time
from typing import List, Tuple

import gymnasium as gym
from gymnasium.wrappers.common import OrderEnforcing
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Wall, Ball
from minigrid.minigrid_env import MiniGridEnv

from logger.logger import AppLogger


class WareHouseEnv(MiniGridEnv):
    def __init__(
        self,
        size: int = 16,
        agent_start_pos: tuple[int, int] = (1, 1),
        agent_start_dir: int = 0,
        max_steps: int | None = None,
        num_obstacles: int = 2,
        obstacle_positions: List[Tuple[int, int]] | None = None,
        **kwargs,
    ) -> None:
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.num_obstacles = num_obstacles

        # Fixed obstacle start positions if not provided
        self.initial_obstacle_positions = obstacle_positions or [(2, 6), (10, 10)]

        # Will be reset each episode
        self.obstacles: List[dict] = []
        self.goal_pos: Tuple[int, int] | None = None

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
        self.goal_pos = (width - 2, height - 2)
        self.put_obj(Goal(), *self.goal_pos)

        # Place dynamic obstacles
        self._place_dynamic_obstacles()

        # Place agent
        self._place_agent_at_starting_position()

        self.mission = "navigate the warehouse, avoid moving obstacles, and reach the goal"

    def _create_grid_world(self, width: int, height: int) -> None:
        self.grid = Grid(width=width, height=height)
        self.grid.wall_rect(x=0, y=0, w=width, h=height)

    def _add_grid_elements(self, width: int, height: int) -> None:
        """
        Create a warehouse-style layout with vertical shelf aisles.
        Gaps allow the agent and obstacles to move between aisles.
        """
        shelf_columns = [3, 6, 9, 12]
        crossing_rows = [4, 8, 12]

        for col in shelf_columns:
            for row in range(1, height - 1):
                if row not in crossing_rows:
                    self.grid.set(i=col, j=row, v=Wall())

    def _place_dynamic_obstacles(self) -> None:
        """
        Add moving obstacles to the warehouse.
        Each obstacle moves horizontally and bounces when blocked.
        """
        self.obstacles = []

        for idx in range(self.num_obstacles):
            if idx < len(self.initial_obstacle_positions):
                pos = self.initial_obstacle_positions[idx]
            else:
                pos = (2 + idx, 2 + idx)

            x, y = pos

            # Safety check so obstacles do not start inside walls or on the goal
            if self.grid.get(x, y) is not None:
                continue
            if self.goal_pos is not None and (x, y) == self.goal_pos:
                continue
            if (x, y) == self.agent_start_pos:
                continue

            obstacle_obj = Ball(color="red")
            self.grid.set(x, y, obstacle_obj)

            self.obstacles.append(
                {
                    "pos": (x, y),
                    "dir": 1,          # +1 means move right, -1 means move left
                    "obj": obstacle_obj,
                }
            )

    def _place_agent_at_starting_position(self) -> None:
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

    @staticmethod
    def _gen_mission() -> str:
        return "navigate the warehouse and reach the goal"

    def _move_obstacles(self) -> None:
        """
        Move each obstacle one step horizontally.
        If blocked by a wall or object, reverse direction.
        """
        for obstacle in self.obstacles:
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
            if self.goal_pos == (old_x, old_y):
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

        if self.goal_pos is not None and (x, y) == self.goal_pos:
            return False

        cell_obj = self.grid.get(x, y)
        if cell_obj is not None:
            return False

        return True

    def _agent_hits_obstacle(self) -> bool:
        return any(obstacle["pos"] == self.agent_pos for obstacle in self.obstacles)

    def step(self, action):
        """
        One environment step:
        1. Move agent using MiniGrid logic
        2. Check collision with obstacle
        3. Move obstacles
        4. Check collision again
        5. Apply custom reward shaping
        """
        obs, reward, terminated, truncated, info = super().step(action)

        # Small step penalty to encourage efficiency
        reward -= 0.01

        # Case 1: agent moves into obstacle
        if self._agent_hits_obstacle():
            reward = -1.0
            terminated = True
            info["collision"] = True
            return obs, reward, terminated, truncated, info

        # If goal already reached, do not move obstacles
        if terminated or truncated:
            return obs, reward, terminated, truncated, info

        # Move obstacles after the agent acts
        self._move_obstacles()

        # Refresh observation after obstacle movement
        obs = self.gen_obs()

        # Case 2: obstacle moves into agent
        if self._agent_hits_obstacle():
            reward = -1.0
            terminated = True
            info["collision"] = True
            return obs, reward, terminated, truncated, info

        info["collision"] = False
        return obs, reward, terminated, truncated, info

    def randomly_navigate_custom_grid_world(self) -> None:
        environment_obj: OrderEnforcing = WareHouseEnv(render_mode="human")
        observation_dict, info_dict = environment_obj.reset(seed=42)

        for _ in range(300):
            action_int = environment_obj.action_space.sample()
            observation_dict, reward_float, terminated_bool, truncated_bool, info_dict = environment_obj.step(
                action_int
            )

            print(
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
        environment_obj: OrderEnforcing = gym.make(
            id="MiniGrid-Empty-16x16-v0",
            render_mode="human"
        )
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