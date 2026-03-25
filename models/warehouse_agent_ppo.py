import math
from dataclasses import dataclass

import numpy as np
import torch
from gymnasium import Env
from torch import Tensor
from torch.distributions import MultivariateNormal

from logger.logger import AppLogger
from models.actor_network import ActorNetwork
from utils.constants import Constants
from warehouse_env.warehouse_env import WareHouseEnv


@dataclass
class StepDataObj:
    observation_dict: dict
    action_int: int
    action_str: str
    reward_float: float
    is_done: bool
    info_dict: dict


class ExponentialGreedyDecayScheduler:
    def __init__(self, value_from: float, value_to: float, num_steps: int):
        self.value_from = value_from
        self.value_to = value_to
        self.num_steps = num_steps

        self.a = value_from
        self.b = math.log(value_to / value_from) / (num_steps - 1)

    def get_epsilon_value(self, current_time_step: int) -> float:
        if current_time_step <= 0:
            return self.value_from

        if current_time_step >= self.num_steps - 1:
            return self.value_to

        value = self.a * math.exp(self.b * current_time_step)

        return value


class WareHouseAgentPPO:

    def __init__(self) -> None:
        self._gamma: float = 0.95
        self._device = self._get_device()
        self._total_time_steps: int = 10_000
        self._time_steps_per_batch: int = 4_000
        self._num_updates_per_iteration: int = 5
        self._max_time_steps_per_episode: int = 2_000
        self._environment_obj: Env = WareHouseEnv(render_mode=None)
        self._action_dimensions = self._environment_obj.action_space.shape[0]
        self._observation_dimensions = self._environment_obj.observation_space.shape[0]
        self._actor_network: ActorNetwork = ActorNetwork(input_dimensions=self._observation_dimensions,
                                                         output_dimensions=self._action_dimensions,
                                                         device=self._device
                                                         )
        # TODO: Create critic network
        self._critic_network = None
        self._exponential_greedy_decay_scheduler: ExponentialGreedyDecayScheduler = ExponentialGreedyDecayScheduler(
            value_from=1.0,
            value_to=0.05,
            num_steps=3_000
        )

        # NOTE: Standard Deviation 0.5 arbitrarily.
        self._covariance_variable = torch.full(size=(self._action_dimensions,), fill_value=0.5)
        self._covariance_matrix = torch.diagonal(input=self._covariance_variable)

        self._logger = AppLogger.get_logger(self.__class__.__name__)

    def evaluate_agent(self, batch_observation_tensor: Tensor, batch_actions_tensor: Tensor) -> tuple[Tensor, Tensor]:
        v_tensor: Tensor = self._critic_network(batch_observation_tensor).squeeze()

        mean_value = self._actor_network(batch_observation_tensor)
        multivariate_normal_distribution: MultivariateNormal = MultivariateNormal(loc=mean_value,
                                                                                  covariance_matrix=self._covariance_matrix)
        log_probabilities_tensor: Tensor = multivariate_normal_distribution.log_prob(value=batch_actions_tensor)

        return v_tensor, log_probabilities_tensor

    def train_agent(self) -> None:
        current_time_step: int = 0

        while current_time_step <= self._total_time_steps:
            batch_observation_tensor, batch_actions_tensor, batch_rewards_tensor, batch_length_tensor = self._rollout()

            # NOTE: Calculates V from the critic_network, and the log probabilities for to minimize the following formula:
            # FORMULA: π_theta(a,s) / π_theta_k(a,s)
            # Old Policy - π_theta(a,s)
            # New Policy - π_theta_k(a,s)
            v_tensor, log_probabilities_tensor = self.evaluate_agent(batch_observation_tensor=batch_observation_tensor,
                                                                     batch_actions_tensor=batch_actions_tensor)

            # NOTE: Calculates the advantage tensor
            advantage_value_tensor: Tensor = self._get_normalized_advantage_value(
                batch_rewards_tensor=batch_rewards_tensor,
                v_tensor=v_tensor)

            for _ in range(0, self._num_updates_per_iteration):
                # TODO: STOPPED HERE -
                pass


    def _get_normalized_advantage_value(self, batch_rewards_tensor, v_tensor) -> Tensor:

        advantage_value_tensor: Tensor = batch_rewards_tensor - v_tensor.detach()

        numerator_value = advantage_value_tensor - advantage_value_tensor.mean()
        denominator_value = advantage_value_tensor.std() + 1e-10

        return numerator_value / denominator_value

    def _rollout(self) -> tuple:

        current_time_step: int = 0
        batch_length_list: list[int] = []
        batch_actions_list: list[int] = []
        batch_observation_list: list[dict] = []
        batch_rewards_list: list[list[float]] = []
        batch_log_probability_list: list[float] = []

        while current_time_step < self._time_steps_per_batch:

            episode_num_value: int = 0
            episode_rewards: list[float] = []
            observation_dict, info_dict = self._environment_obj.reset(seed=42)

            for episode_num in range(0, self._max_time_steps_per_episode):

                current_time_step += 1
                batch_observation_list.append(observation_dict)

                action_int, log_probability = self._get_action(observation_dict=observation_dict,
                                                               current_time_step=current_time_step)

                observation_dict, reward, terminated_bool, truncated_bool, info_dict = self._environment_obj.step(
                    action_int
                )
                is_done: bool = terminated_bool or truncated_bool

                episode_rewards.append(float(reward))
                batch_actions_list.append(action_int)
                batch_log_probability_list.append(log_probability)
                episode_num_value = episode_num

                if is_done:
                    observation_dict, info_dict = self._environment_obj.reset()

            batch_length_list.append(episode_num_value + 1)
            batch_rewards_list.append(episode_rewards)

        batch_observation_tensor: Tensor = torch.tensor(data=batch_observation_list, dtype=torch.dict,
                                                        device=self._device)
        batch_actions_tensor: Tensor = torch.tensor(data=batch_actions_list, dtype=torch.int, device=self._device)
        batch_rewards_tensor: Tensor = self._get_rewards_tensor(batch_rewards_list=batch_rewards_list)
        batch_length_tensor: Tensor = torch.tensor(data=batch_length_list, dtype=torch.int, device=self._device)

        self._environment_obj.close()

        return batch_observation_tensor, batch_actions_tensor, batch_rewards_tensor, batch_length_tensor

    def _get_rewards_tensor(self, batch_rewards_list: list[list[float]]) -> Tensor:

        batch_rewards: list[float] = []

        for episode_rewards_list in batch_rewards_list:

            discounted_reward: float = 0.0

            for reward_value in reversed(episode_rewards_list):
                discounted_reward = reward_value + discounted_reward * self._gamma
                # Places each newly added value to the beginning of the list
                # because we are working from the back of the episode_rewards_list
                batch_rewards.insert(0, discounted_reward)

        batch_rewards_tensor = torch.tensor(data=batch_rewards, dtype=torch.list, device=self._device)

        return batch_rewards_tensor

    # TODO: Should return action_int and log_probabilities
    def _get_action(self, observation_dict: dict, current_time_step: int) -> int:

        action_int: int = int(math.inf)
        mean_value = self._actor_network(observation_dict)
        multivariate_normal_distribution: MultivariateNormal = MultivariateNormal(loc=mean_value,
                                                                                  covariance_matrix=self._covariance_matrix)

        epsilon_value: float = self._exponential_greedy_decay_scheduler.get_epsilon_value(
            current_time_step=current_time_step)

        if np.random.rand() < epsilon_value:
            action_int = self._environment_obj.action_space.sample()
            # TODO: Should have a return statement in this conditional with
            #       the log_probabilities returns with the action_int

        else:
            action_int = np.argmax(a=self._actor_network(action_int))
            # TODO: Should have a return statement in this conditional with
            #       the log_probabilities returns with the action_int

        return action_int

    def simulate_agent_environment_navigation(self, num_trial_runs: int) -> None:

        for num_trial in range(0, num_trial_runs):

            observation_dict, info_dict = self._environment_obj.reset(seed=42)

            is_done: bool = False
            episode_step_counter: int = 0
            rewards_list: list[float] = []
            action_taken_list: list[int] = []

            while not is_done or episode_step_counter >= self._max_time_steps_per_episode:
                episode_step_counter += 1
                action_int = self._environment_obj.action_space.sample()
                observation_dict, reward, terminated_bool, truncated_bool, info_dict = self._environment_obj.step(
                    action_int
                )

                action_str: str = Constants.ACTION_SPACE_MAPPING_DICT.get(action_int, "")
                is_done = terminated_bool or truncated_bool
                reward_float: float = float(reward)

                step_data_obj: StepDataObj = StepDataObj(observation_dict=observation_dict,
                                                         action_int=action_int,
                                                         action_str=action_str,
                                                         reward_float=reward_float,
                                                         is_done=is_done,
                                                         info_dict=info_dict)

                self.display_step_information(step_data_obj=step_data_obj)

                reward_float: float = float(reward)
                rewards_list.append(reward_float)
                action_taken_list.append(action_int)


                if is_done:
                    self._logger.info(f"Trial Number: {num_trial + 1}")
                    self._logger.info(f"Reached goal state in {episode_step_counter:,} steps")
                    self._logger.info("=" * 100)
                    observation_dict, info_dict = self._environment_obj.reset()

        self._environment_obj.close()

    def display_step_information(self, step_data_obj: StepDataObj) -> None:

        self._logger.info(
            f"action_int={step_data_obj.action_int}, "
            f"reward={step_data_obj.reward_float:.2f}, "
            f"action_str={step_data_obj.action_str}, "
            f"is_done={step_data_obj.is_done}, "
            f"info={step_data_obj.info_dict}"
        )

    def _get_device(self):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.mps.is_available():
            return torch.device("mps")

        return torch.device("cpu")
