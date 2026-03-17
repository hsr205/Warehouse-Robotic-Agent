import time

import gymnasium as gym
from gymnasium.wrappers.common import OrderEnforcing
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Wall
from minigrid.minigrid_env import MiniGridEnv

from logger.logger import AppLogger


class WareHouseEnv(MiniGridEnv):

    def __init__(self,
                 size: int = 16,
                 agent_start_pos: tuple[int, int] = (1, 1),
                 agent_start_dir: int = 0,
                 max_steps: int | None = None,
                 **kwargs, ) -> None:
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

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

        self._add_grid_elements(height=height)

        # 4. Add the goal
        self.put_obj(Goal(), width - 2, height - 2)

        # 5. Place the agent
        self._place_agent_at_starting_position()

        # 6. Set the mission text shown by MiniGrid
        self.mission = "reach the goal"

    def _create_grid_world(self, width: int, height: int) -> None:
        self.grid = Grid(width=width, height=height)
        self.grid.wall_rect(x=0, y=0, w=width, h=height)

    def _add_grid_elements(self, height:int) -> None:
        # 3. Add one internal wall as an example
        for row_int in range(2, height - 2):
            self.grid.set(i=5, j=row_int, v=Wall())

    def _place_agent_at_starting_position(self) -> None:
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

    @staticmethod
    def _gen_mission() -> str:
        return "reach the goal"

    def randomly_navigate_custom_grid_world(self) -> None:
        environment_obj: OrderEnforcing = WareHouseEnv(render_mode="human")
        observation_dict, info_dict = environment_obj.reset(seed=42)

        for _ in range(200):
            action_int = environment_obj.action_space.sample()
            observation_dict, reward_float, terminated_bool, truncated_bool, info_dict = environment_obj.step(
                action_int)

            time.sleep(0.1)

            if terminated_bool or truncated_bool:
                observation_dict, info_dict = environment_obj.reset()

        environment_obj.close()

    def randomly_navigate_empty_grid_world(self) -> None:
        environment_obj: OrderEnforcing = gym.make(id="MiniGrid-Empty-16x16-v0", render_mode="human")
        observation_dict, info_dict = environment_obj.reset(seed=42)

        for _ in range(200):
            action_int = environment_obj.action_space.sample()
            observation_dict, reward_float, terminated_bool, truncated_bool, info_dict = environment_obj.step(
                action_int)

            time.sleep(0.1)

            if terminated_bool or truncated_bool:
                observation_dict, info_dict = environment_obj.reset()

        environment_obj.close()
