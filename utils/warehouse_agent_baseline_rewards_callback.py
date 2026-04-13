from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm


class WarehouseAgentBaselineRewardsCallBack(BaseCallback):

    def __init__(self, total_time_steps: int, algorithm_name_str:str, verbose: int = 0) -> None:
        super().__init__(verbose=verbose)

        self.training_time_steps: list[int] = []
        self.training_rewards: list[float] = []

        self.training_episode_numbers: list[int] = []
        self.training_episode_rewards: list[float] = []
        self._total_time_steps: int = total_time_steps
        self._algorithm_name_str:str = algorithm_name_str

        self._current_episode_reward: float = 0.0
        self._episode_counter: int = 0
        self._progress_bar: tqdm | None = None

    def _on_training_start(self) -> None:
        self._progress_bar = tqdm(
            total=self._total_time_steps,
            desc="Training Baseline Agent",
            dynamic_ncols=True,
        )

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards")
        dones = self.locals.get("dones")

        if rewards is None or dones is None:
            return True

        for reward_value, is_done in zip(rewards, dones):
            current_time_step: int = len(self.training_time_steps) + 1

            self.training_time_steps.append(current_time_step)
            self.training_rewards.append(float(reward_value))

            self._current_episode_reward += float(reward_value)

            if bool(is_done):
                self._episode_counter += 1
                self.training_episode_numbers.append(self._episode_counter)
                self.training_episode_rewards.append(self._current_episode_reward)
                self._current_episode_reward = 0.0

        if self._progress_bar is not None:
            current_n: int = self._progress_bar.n
            target_n: int = self.num_timesteps
            increment_value: int = target_n - current_n

            if increment_value > 0:
                self._progress_bar.update(increment_value)

        return True

    def _on_training_end(self) -> None:
        if self._progress_bar is not None:
            self._progress_bar.close()
            self._progress_bar = None