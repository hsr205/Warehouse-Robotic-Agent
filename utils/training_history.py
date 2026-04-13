from dataclasses import dataclass


@dataclass
class TrainingHistory:
    algorithm_name: str
    training_time_steps: list[int]
    training_rewards: list[float]
    training_episode_numbers: list[int]
    training_episode_rewards: list[float]
