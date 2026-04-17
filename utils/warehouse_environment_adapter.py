from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Discrete

from warehouse_env.warehouse_env import WareHouseEnv
from warehouse_env.warehouse_env_2 import WareHouseEnv2
from warehouse_env.warehouse_env_3 import WareHouseEnv3


class WarehouseEnvironmentAdapter(gym.Env):
    metadata_dict: dict[str, list] = {"render_modes": []}

    def __init__(self, environment_obj: WareHouseEnv | WareHouseEnv2 | WareHouseEnv3, ) -> None:

        super().__init__()

        self.action_space = Discrete(4)
        self._environment_obj = environment_obj

        observation_dict, _ = self._environment_obj.reset(seed=42)
        image_array = self._extract_image_array(observation_dict=observation_dict)

        self.observation_space = Box(
            low=0,
            high=255,
            shape=image_array.shape,
            dtype=np.uint8,
        )

    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        observation_dict, info_dict = self._environment_obj.reset(seed=seed)

        image_array: np.ndarray = self._extract_image_array(observation_dict=observation_dict)

        result_tuple: tuple[np.ndarray, dict[str, Any]] = image_array, info_dict

        return result_tuple

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        observation_dict, reward_value, is_terminated, is_truncated, info_dict = self._environment_obj.step(action)

        image_array: np.ndarray = self._extract_image_array(observation_dict=observation_dict)

        result_tuple: tuple[np.ndarray, float, bool, bool, dict[str, Any]] = image_array, float(reward_value), bool(
            is_terminated), bool(is_truncated), info_dict

        return result_tuple

    def _extract_image_array(self, observation_dict: dict[str, Any]) -> np.ndarray:
        image_array: np.ndarray = observation_dict.get("image")

        if image_array is None:
            raise ValueError("Expected observation_dict to contain key 'image'.")

        image_array: np.ndarray = np.asarray(image_array)

        # if image_array.dtype != np.uint8:
        #     image_array = image_array.astype(np.uint8)

        return image_array

    def close(self) -> None:
        self._environment_obj.close()

    def render(self) -> None:
        if hasattr(self._environment_obj, "render"):
            self._environment_obj.render()
