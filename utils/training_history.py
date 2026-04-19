from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(slots=True)
class TrainingHistory:
    algorithm_name: str
    environment_name: str
    training_time_steps: list[int]
    training_rewards: list[float]
    training_episode_numbers: list[int]
    training_episode_rewards: list[float]
    training_episode_time_steps: list[int]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def save_to_json(
            self,
            file_path: Path,
            compact_time_step_history: bool = False,
            max_time_step_points: int = 25_000,
    ) -> None:
        history_to_save: TrainingHistory = self

        if compact_time_step_history:
            history_to_save = self.get_compacted_copy(max_time_step_points=max_time_step_points)

        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(
            json.dumps(history_to_save.to_dict(), indent=2),
            encoding="utf-8",
        )

    def get_compacted_copy(self, max_time_step_points: int = 25_000) -> "TrainingHistory":
        compacted_time_steps_list, compacted_rewards_list = self._compact_time_step_history(
            max_time_step_points=max_time_step_points
        )

        return TrainingHistory(
            algorithm_name=self.algorithm_name,
            environment_name=self.environment_name,
            training_time_steps=compacted_time_steps_list,
            training_rewards=compacted_rewards_list,
            training_episode_numbers=self.training_episode_numbers.copy(),
            training_episode_rewards=self.training_episode_rewards.copy(),
            training_episode_time_steps=self.training_episode_time_steps.copy(),
        )

    @classmethod
    def from_dict(cls, data_dict: dict[str, object]) -> "TrainingHistory":
        return cls(
            algorithm_name=str(data_dict["algorithm_name"]),
            environment_name=str(data_dict["environment_name"]),
            training_time_steps=[int(value) for value in data_dict["training_time_steps"]],
            training_rewards=[float(value) for value in data_dict["training_rewards"]],
            training_episode_numbers=[int(value) for value in data_dict["training_episode_numbers"]],
            training_episode_rewards=[float(value) for value in data_dict["training_episode_rewards"]],
            training_episode_time_steps=[
                int(value)
                for value in data_dict.get("training_episode_time_steps", [])
            ],
        )

    @classmethod
    def load_from_json(cls, file_path: Path) -> "TrainingHistory":
        return cls.from_dict(json.loads(file_path.read_text(encoding="utf-8")))

    def _compact_time_step_history(self, max_time_step_points: int) -> tuple[list[int], list[float]]:
        if len(self.training_time_steps) <= max_time_step_points:
            return self.training_time_steps.copy(), self.training_rewards.copy()

        chunk_size: int = max(1, len(self.training_time_steps) // max_time_step_points)

        compacted_time_steps_list: list[int] = []
        compacted_rewards_list: list[float] = []

        for start_index in range(0, len(self.training_time_steps), chunk_size):
            end_index = min(start_index + chunk_size, len(self.training_time_steps))

            time_step_chunk_list = self.training_time_steps[start_index:end_index]
            reward_chunk_list = self.training_rewards[start_index:end_index]

            if len(time_step_chunk_list) == 0:
                continue

            compacted_time_steps_list.append(int(time_step_chunk_list[-1]))
            compacted_rewards_list.append(float(sum(reward_chunk_list) / len(reward_chunk_list)))

        if compacted_time_steps_list[-1] != self.training_time_steps[-1]:
            compacted_time_steps_list.append(int(self.training_time_steps[-1]))
            compacted_rewards_list.append(float(self.training_rewards[-1]))

        return compacted_time_steps_list, compacted_rewards_list
