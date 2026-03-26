import numpy as np
import torch
from gymnasium import Env
from torch import nn, Tensor
from torch.distributions import MultivariateNormal
from torch.optim import Adam

from logger.logger import AppLogger
from models.actor_network import ActorNetwork
from models.exponential_greedy_decay_scheduler import ExponentialGreedyDecayScheduler
from models.step_data_obj import StepDataObj
from utils.constants import Constants
from warehouse_env.warehouse_env import WareHouseEnv


class WareHouseAgentPPO:

    def __init__(self) -> None:
        # NOTE: CLIP acts as a threshold for making sure our policy
        #       does not change too dramatically when conducting SGA
        self._clip: float = 0.2
        self._gamma: float = 0.95
        self._device = self._get_device()
        self._learning_rate: float = 0.005
        self._total_time_steps: int = 10_000
        self._time_steps_per_batch: int = 4_000
        self._num_updates_per_iteration: int = 5
        self._num_training_steps: int = 1_000_000
        self._max_time_steps_per_episode: int = 2_000
        self._environment_obj: Env = WareHouseEnv(render_mode=None)
        self._action_dimensions = self._environment_obj.action_space.n
        self._observation_dimensions = self._environment_obj.observation_space.get("direction").n

        self._actor_network: ActorNetwork = ActorNetwork(input_dimensions=self._observation_dimensions,
                                                         output_dimensions=self._action_dimensions,
                                                         device=self._device
                                                         )
        # TODO: Create separate critic network
        self._critic_network: ActorNetwork = ActorNetwork(input_dimensions=self._observation_dimensions,
                                                          output_dimensions=self._action_dimensions,
                                                          device=self._device
                                                          )

        self._actor_network_optimizer: Adam = Adam(params=self._actor_network.parameters(),
                                                   lr=self._learning_rate)

        self._critic_network_optimizer: Adam = Adam(params=self._critic_network.parameters(),
                                                    lr=self._learning_rate)

        self._exponential_greedy_decay_scheduler: ExponentialGreedyDecayScheduler = ExponentialGreedyDecayScheduler(
            value_from=1.0,
            value_to=0.05,
            num_steps=3_000
        )

        # NOTE: Standard deviation set to 0.5 arbitrarily.
        self._covariance_variable = torch.full(size=(self._action_dimensions,), fill_value=0.5)
        self._covariance_matrix = torch.diag(input=self._covariance_variable)

        self._logger = AppLogger.get_logger(self.__class__.__name__)

    def _evaluate_agent(self, batch_observation_tensor: Tensor, batch_actions_tensor: Tensor) -> tuple[Tensor, Tensor]:
        v_tensor: Tensor = self._critic_network(batch_observation_tensor).squeeze()

        mean_value = self._actor_network(batch_observation_tensor)
        multivariate_normal_distribution: MultivariateNormal = MultivariateNormal(loc=mean_value,
                                                                                  covariance_matrix=self._covariance_matrix)
        log_probabilities_tensor: Tensor = multivariate_normal_distribution.log_prob(value=batch_actions_tensor)

        return v_tensor, log_probabilities_tensor

    def train_agent(self) -> None:
        current_time_step: int = 0

        # TODO: Implement the following progress_bar
        # progress_bar: tqdm = tqdm.trange(self._num_training_steps)

        self._logger.info("Initializing Agent PPO Training")
        self._logger.info("=" * 100)

        while current_time_step <= self._total_time_steps:

            if current_time_step % 100 == 0:
                self._logger.info(f"Current Time Step: {current_time_step + 1}")
                self._logger.info("=" * 100)

            batch_observation_tensor, batch_actions_tensor, batch_rewards_tensor, batch_length_tensor, batch_log_probability_tensor = self._rollout()

            # NOTE: Calculates V from the critic_network, and the log probabilities for to minimize the following formula:
            # FORMULA: π_theta(a_t,s_t) / π_theta_k(a_t,s_t)
            # Old Policy - π_theta(a,s)
            # New Policy - π_theta_k(a,s)
            v_tensor, _ = self._evaluate_agent(batch_observation_tensor=batch_observation_tensor,
                                               batch_actions_tensor=batch_actions_tensor)

            # NOTE: Calculates the advantage tensor
            advantage_value_tensor: Tensor = self._get_normalized_advantage_value(
                batch_rewards_tensor=batch_rewards_tensor,
                v_tensor=v_tensor)

            for _ in range(0, self._num_updates_per_iteration):
                v_tensor, current_log_probabilities_tensor = self._evaluate_agent(
                    batch_observation_tensor=batch_observation_tensor,
                    batch_actions_tensor=batch_actions_tensor)

                # NOTE: This ratio is simply - π_theta(a_t | s_t) / π_theta_k(a_t | s_t)
                ratios_tensor: Tensor = torch.exp(
                    input=(current_log_probabilities_tensor - batch_log_probability_tensor))

                surrogate_loss_tensor_1: Tensor = ratios_tensor * advantage_value_tensor

                # NOTE: The following prevents the ratio from take too large a step
                #       This is one if not thee core principle of PPO
                lower_bound_clip_value: float = 1 - self._clip
                upper_bound_clip_value: float = 1 + self._clip
                surrogate_loss_tensor_2: Tensor = torch.clamp(ratios_tensor, lower_bound_clip_value,
                                                              upper_bound_clip_value) * advantage_value_tensor
                # NOTE: Maximizes loss through negation
                #       this will be optimized by Adam later and will improve performance
                actor_network_loss_tensor: Tensor = (
                    -torch.min(surrogate_loss_tensor_1, surrogate_loss_tensor_2)).mean()

                # NOTE: Zeroes out the gradient
                #       Conducts a backward propagation of the calculated loss
                #       Leverages the optimizer to take a step (determined by the learning rate) forward
                self._actor_network_optimizer.zero_grad()

                # NOTE: Retains the computation graph for later back propagation
                actor_network_loss_tensor.backward(retain_graph=True)
                self._actor_network_optimizer.step()

                critic_network_loss_tensor: Tensor = nn.MSELoss()(v_tensor, batch_rewards_tensor)

                self._critic_network_optimizer.zero_grad()
                critic_network_loss_tensor.backward()
                self._critic_network_optimizer.step()

            current_time_step += np.sum(batch_length_tensor)

    def _get_normalized_advantage_value(self, batch_rewards_tensor, v_tensor) -> Tensor:

        advantage_value_tensor: Tensor = batch_rewards_tensor - v_tensor.detach()

        numerator_value = advantage_value_tensor - advantage_value_tensor.mean()
        denominator_value = advantage_value_tensor.std() + 1e-10

        return numerator_value / denominator_value

    def _rollout(self) -> tuple:

        self._logger.info(f"Inside _rollout() method")

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

            self._logger.info(f"Inside _rollout() method - while loop")

            for episode_num in range(0, self._max_time_steps_per_episode):

                self._logger.info(f"Inside _rollout() method - for loop")

                current_time_step += 1
                batch_observation_list.append(observation_dict)

                action_int, log_probability = self._get_action(observation_dict=observation_dict,
                                                               current_time_step=current_time_step)

                self._logger.info(f"Breaking in method: {__method__}")
                self._logger.info(f"type(action_int) = {type(action_int)}")
                self._logger.info(f"type(log_probability) = {type(log_probability)}")

                break

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

            break

        batch_observation_tensor: Tensor = torch.tensor(data=batch_observation_list,
                                                        dtype=torch.dict,
                                                        device=self._device)

        batch_actions_tensor: Tensor = torch.tensor(data=batch_actions_list,
                                                    dtype=torch.int,
                                                    device=self._device)

        batch_rewards_tensor: Tensor = self._get_rewards_tensor(batch_rewards_list=batch_rewards_list)

        batch_length_tensor: Tensor = torch.tensor(data=batch_length_list,
                                                   dtype=torch.int,
                                                   device=self._device)

        batch_log_probability_tensor: Tensor = torch.tensor(data=batch_log_probability_list,
                                                            dtype=torch.list,
                                                            device=self._device)

        self._environment_obj.close()

        return batch_observation_tensor, batch_actions_tensor, batch_rewards_tensor, batch_length_tensor, batch_log_probability_tensor

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

    def _get_action(self, observation_dict: dict, current_time_step: int) -> tuple[Tensor, Tensor]:

        observation_image_array:np.ndarray = observation_dict.get("image")

        observation_tensor:Tensor = torch.tensor(data=observation_image_array, dtype=torch.float32, device=self._device)


        mean_value = self._actor_network(observation_tensor)

        # NOTE: Creates a normal distribution across all variables (actions)
        multivariate_normal_distribution: MultivariateNormal = MultivariateNormal(loc=mean_value,
                                                                                  covariance_matrix=self._covariance_matrix)


        action_tensor: Tensor = torch.tensor(data=[], device=self._device)
        log_probabilities_tensor: Tensor = multivariate_normal_distribution.log_prob(value=action_tensor)

        epsilon_value: float = self._exponential_greedy_decay_scheduler.get_epsilon_value(
            current_time_step=current_time_step)


        # NOTE: Explore using e-greedy decay
        if np.random.rand() < epsilon_value:
            action_tensor = multivariate_normal_distribution.sample()
            log_probabilities_tensor = multivariate_normal_distribution.log_prob(value=action_tensor)

            return action_tensor.detach().numpy(), log_probabilities_tensor.detach()

        # NOTE: Exploit using argmax of all possible actions
        action_tensor = np.argmax(a=self._actor_network(action_tensor))

        self._logger.info("End of _get_action() method")

        return action_tensor, log_probabilities_tensor

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
