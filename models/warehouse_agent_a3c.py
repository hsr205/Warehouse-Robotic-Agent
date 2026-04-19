import os
import queue
import traceback
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
from gymnasium import Env
from gymnasium.spaces import Discrete
from torch import Tensor
from torch.distributions import Categorical
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from tqdm import tqdm

from logger.logger import AppLogger
from models.actor_network import ActorNetwork
from models.critic_network import CriticNetwork
from utils.model_plotting import ModelPlotting
from utils.training_history import TrainingHistory
from warehouse_env.warehouse_env import WareHouseEnv
from warehouse_env.warehouse_env_2 import WareHouseEnv2
from warehouse_env.warehouse_env_3 import WareHouseEnv3


WarehouseEnvironmentClass = type[WareHouseEnv] | type[WareHouseEnv2] | type[WareHouseEnv3]


class SharedAdam(Adam):
    def __init__(self, params, lr: float) -> None:
        super().__init__(params=params, lr=lr)

        for param_group in self.param_groups:
            for parameter in param_group["params"]:
                state = self.state[parameter]
                state["step"] = torch.zeros(1)
                state["exp_avg"] = torch.zeros_like(parameter.data)
                state["exp_avg_sq"] = torch.zeros_like(parameter.data)

        self.share_memory()

    def share_memory(self) -> None:
        for param_group in self.param_groups:
            for parameter in param_group["params"]:
                state = self.state[parameter]
                state["step"].share_memory_()
                state["exp_avg"].share_memory_()
                state["exp_avg_sq"].share_memory_()


@dataclass(slots=True)
class A3CWorkerConfig:
    gamma: float
    entropy_coefficient: float
    critic_coefficient: float
    max_grad_norm: float
    action_dimensions: int
    worker_rollout_steps: int
    max_global_time_steps: int
    max_time_steps_per_episode: int
    base_seed: int
    environment_class: WarehouseEnvironmentClass


def _observation_dict_to_tensor(observation_dict: dict, device: torch.device) -> Tensor:
    observation_image_array: np.ndarray = observation_dict.get("image")
    return torch.tensor(
        data=observation_image_array,
        dtype=torch.float32,
        device=device
    )


def _get_batch_observations_tensor(batch_observation_list: list[dict], device: torch.device) -> Tensor:
    batch_observation_image_list: list[np.ndarray] = []

    for observation_dict in batch_observation_list:
        observation_image_array: np.ndarray = observation_dict.get("image")
        batch_observation_image_list.append(observation_image_array)

    batch_observation_image_array: np.ndarray = np.array(batch_observation_image_list)

    batch_observation_tensor: Tensor = torch.tensor(
        data=batch_observation_image_array,
        dtype=torch.float32,
        device=device
    )

    return batch_observation_tensor.permute(0, 3, 1, 2)


def _get_discounted_returns_tensor(
        rewards_list: list[float],
        bootstrap_value_float: float,
        gamma: float,
        device: torch.device,
) -> Tensor:
    discounted_returns_list: list[float] = []
    running_return_float: float = bootstrap_value_float

    for reward_value in reversed(rewards_list):
        running_return_float = reward_value + gamma * running_return_float
        discounted_returns_list.append(running_return_float)

    discounted_returns_list.reverse()

    return torch.tensor(
        data=discounted_returns_list,
        dtype=torch.float32,
        device=device
    )


def _copy_gradients_to_global(local_model: torch.nn.Module, global_model: torch.nn.Module) -> None:
    for local_parameter, global_parameter in zip(local_model.parameters(), global_model.parameters()):
        if local_parameter.grad is None:
            continue

        if global_parameter.grad is None:
            global_parameter._grad = local_parameter.grad.detach().clone()
        else:
            global_parameter.grad.copy_(local_parameter.grad)


def _sync_local_with_global(local_model: torch.nn.Module, global_model: torch.nn.Module) -> None:
    local_model.load_state_dict(global_model.state_dict())


def _run_a3c_worker(
        worker_id: int,
        worker_config: A3CWorkerConfig,
        global_actor_network: ActorNetwork,
        global_critic_network: CriticNetwork,
        global_actor_optimizer: SharedAdam,
        global_critic_optimizer: SharedAdam,
        optimizer_lock,
        metrics_queue,
        global_time_step_value,
        global_episode_value,
        global_update_value,
) -> None:
    torch.set_num_threads(1)

    worker_logger = AppLogger.get_logger(f"A3CWorker{worker_id}")
    worker_device = torch.device("cpu")

    torch.manual_seed(worker_config.base_seed + worker_id)
    np.random.seed(worker_config.base_seed + worker_id)

    local_actor_network = ActorNetwork(
        output_dimensions=worker_config.action_dimensions,
        device=worker_device
    )
    local_critic_network = CriticNetwork(
        device=worker_device
    )

    _sync_local_with_global(local_actor_network, global_actor_network)
    _sync_local_with_global(local_critic_network, global_critic_network)

    environment_obj: Env = worker_config.environment_class(render_mode=None)
    metrics_queue.put(("worker_started", worker_id, worker_config.base_seed + worker_id))

    local_episode_number: int = 0
    observation_dict, info_dict = environment_obj.reset(seed=worker_config.base_seed + worker_id)

    current_episode_reward: float = 0.0
    current_episode_length: int = 0

    try:
        while True:
            with global_time_step_value.get_lock():
                if global_time_step_value.value >= worker_config.max_global_time_steps:
                    break

            batch_observations_list: list[dict] = []
            batch_actions_list: list[int] = []
            batch_rewards_list: list[float] = []
            rollout_step_metrics_list: list[tuple[int, float]] = []
            rollout_episode_rewards_list: list[float] = []
            rollout_episode_lengths_list: list[int] = []
            rollout_episode_successes_list: list[int] = []

            is_terminated: bool = False
            is_truncated: bool = False
            episode_cutoff_reached: bool = False
            next_observation_dict: dict = observation_dict

            for _ in range(worker_config.worker_rollout_steps):
                with global_time_step_value.get_lock():
                    if global_time_step_value.value >= worker_config.max_global_time_steps:
                        break

                observation_tensor: Tensor = _observation_dict_to_tensor(
                    observation_dict=observation_dict,
                    device=worker_device
                )

                with torch.no_grad():
                    action_probabilities_tensor: Tensor = local_actor_network(observation_tensor)
                    categorical_distribution: Categorical = Categorical(probs=action_probabilities_tensor)
                    action_tensor: Tensor = categorical_distribution.sample()

                action_int: int = int(action_tensor.item())

                next_observation_dict, reward, is_terminated, is_truncated, info_dict = environment_obj.step(action_int)

                batch_observations_list.append(observation_dict)
                batch_actions_list.append(action_int)
                batch_rewards_list.append(float(reward))

                current_episode_reward += float(reward)
                current_episode_length += 1

                with global_time_step_value.get_lock():
                    global_time_step_value.value += 1
                    current_global_time_step: int = int(global_time_step_value.value)

                rollout_step_metrics_list.append((current_global_time_step, float(reward)))

                observation_dict = next_observation_dict

                if is_terminated or is_truncated or current_episode_length >= worker_config.max_time_steps_per_episode:
                    episode_cutoff_reached = current_episode_length >= worker_config.max_time_steps_per_episode
                    with global_episode_value.get_lock():
                        global_episode_value.value += 1
                        current_global_episode: int = int(global_episode_value.value)

                    episode_success: int = int(is_terminated)
                    rollout_episode_rewards_list.append(current_episode_reward)
                    rollout_episode_lengths_list.append(current_episode_length)
                    rollout_episode_successes_list.append(episode_success)

                    metrics_queue.put(
                        (
                            "episode",
                            current_global_episode,
                            current_episode_reward,
                            current_global_time_step,
                            current_episode_length,
                            episode_success,
                        )
                    )

                    local_episode_number += 1
                    observation_dict, info_dict = environment_obj.reset(
                        seed=worker_config.base_seed + worker_id + local_episode_number
                    )
                    current_episode_reward = 0.0
                    current_episode_length = 0
                    break

            if rollout_step_metrics_list:
                metrics_queue.put(
                    (
                        "rollout",
                        worker_id,
                        len(rollout_step_metrics_list),
                        rollout_step_metrics_list,
                    )
                )

            if not batch_observations_list:
                continue

            if is_terminated or is_truncated or episode_cutoff_reached:
                bootstrap_value_float: float = 0.0
            else:
                with torch.no_grad():
                    next_state_value_tensor: Tensor = local_critic_network(
                        _observation_dict_to_tensor(
                            observation_dict=next_observation_dict,
                            device=worker_device
                        )
                    ).squeeze(-1)
                    bootstrap_value_float = float(next_state_value_tensor.item())

            batch_observation_tensor: Tensor = _get_batch_observations_tensor(
                batch_observation_list=batch_observations_list,
                device=worker_device
            )
            batch_actions_tensor: Tensor = torch.tensor(
                data=batch_actions_list,
                dtype=torch.long,
                device=worker_device
            )
            batch_returns_tensor: Tensor = _get_discounted_returns_tensor(
                rewards_list=batch_rewards_list,
                bootstrap_value_float=bootstrap_value_float,
                gamma=worker_config.gamma,
                device=worker_device
            )

            action_probabilities_tensor: Tensor = local_actor_network(batch_observation_tensor)
            categorical_distribution: Categorical = Categorical(probs=action_probabilities_tensor)
            log_probabilities_tensor: Tensor = categorical_distribution.log_prob(batch_actions_tensor)
            entropy_tensor: Tensor = categorical_distribution.entropy().mean()

            state_values_tensor: Tensor = local_critic_network(batch_observation_tensor).squeeze(-1)
            advantage_tensor: Tensor = batch_returns_tensor - state_values_tensor

            actor_loss_tensor: Tensor = -(log_probabilities_tensor * advantage_tensor.detach()).mean()
            actor_loss_tensor = actor_loss_tensor - worker_config.entropy_coefficient * entropy_tensor

            critic_loss_tensor: Tensor = torch.nn.functional.mse_loss(
                state_values_tensor,
                batch_returns_tensor
            )

            total_loss_tensor: Tensor = actor_loss_tensor + worker_config.critic_coefficient * critic_loss_tensor

            local_actor_network.zero_grad()
            local_critic_network.zero_grad()
            total_loss_tensor.backward()

            clip_grad_norm_(local_actor_network.parameters(), max_norm=worker_config.max_grad_norm)
            clip_grad_norm_(local_critic_network.parameters(), max_norm=worker_config.max_grad_norm)

            with optimizer_lock:
                global_actor_optimizer.zero_grad()
                global_critic_optimizer.zero_grad()

                _copy_gradients_to_global(local_actor_network, global_actor_network)
                _copy_gradients_to_global(local_critic_network, global_critic_network)

                global_actor_optimizer.step()
                global_critic_optimizer.step()

                _sync_local_with_global(local_actor_network, global_actor_network)
                _sync_local_with_global(local_critic_network, global_critic_network)

                with global_update_value.get_lock():
                    global_update_value.value += 1
                    current_update_number: int = int(global_update_value.value)

            mean_episode_reward: float = (
                sum(rollout_episode_rewards_list) / len(rollout_episode_rewards_list)
                if rollout_episode_rewards_list else 0.0
            )
            mean_episode_length: float = (
                sum(rollout_episode_lengths_list) / len(rollout_episode_lengths_list)
                if rollout_episode_lengths_list else 0.0
            )
            success_rate: float = (
                sum(rollout_episode_successes_list) / len(rollout_episode_successes_list)
                if rollout_episode_successes_list else 0.0
            )

            metrics_queue.put(
                (
                    "update",
                    current_update_number,
                    worker_id,
                    float(actor_loss_tensor.item()),
                    float(critic_loss_tensor.item()),
                    float(entropy_tensor.item()),
                    current_global_time_step,
                    mean_episode_reward,
                    mean_episode_length,
                    success_rate,
                )
            )

    except Exception as exception:
        worker_logger.error(f"A3C worker {worker_id} crashed: {exception}")
        metrics_queue.put(
            (
                "error",
                worker_id,
                str(exception),
                traceback.format_exc(),
            )
        )
        raise
    finally:
        environment_obj.close()


class WareHouseAgentA3C:
    def __init__(
            self,
            environment_class: WarehouseEnvironmentClass = WareHouseEnv,
            number_of_workers: int | None = None,
            worker_rollout_steps: int = 32,
            max_global_time_steps: int = 1_000_000,
            max_time_steps_per_episode: int = 500,
            checkpoint_interval_updates: int = 5_000,
            resume_from_latest_checkpoint: bool = True,
            resume_checkpoint_filename: str | None = None,
            checkpoint_directory_name: str = "model_weights_a3c",
            plot_directory_name: str = "model_plots_a3c",
            base_seed: int = 42,
    ) -> None:
        self._gamma: float = 0.95
        self._learning_rate: float = 3e-4
        self._entropy_coefficient: float = 0.08
        self._critic_coefficient: float = 0.5
        self._max_grad_norm: float = 0.5
        self._timestamp_string: str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self._base_seed: int = base_seed
        self._environment_class: WarehouseEnvironmentClass = environment_class

        default_worker_count: int = 2
        self._number_of_workers: int = number_of_workers if number_of_workers is not None else default_worker_count
        self._worker_rollout_steps: int = worker_rollout_steps
        self._max_global_time_steps: int = max_global_time_steps
        self._max_time_steps_per_episode: int = max_time_steps_per_episode
        self._checkpoint_interval_updates: int = checkpoint_interval_updates
        self._plot_interval_updates: int = checkpoint_interval_updates * 3
        self._resume_from_latest_checkpoint: bool = resume_from_latest_checkpoint
        self._resume_checkpoint_filename: str | None = resume_checkpoint_filename
        self._checkpoint_directory_name: str = checkpoint_directory_name
        self._plot_directory_name: str = plot_directory_name

        self._logger = AppLogger.get_logger(self.__class__.__name__)
        self._model_plotting: ModelPlotting = ModelPlotting()

        # We keep A3C training on CPU because the global models are shared
        # across worker processes.
        self._device = self._get_device()
        self._action_dimensions: int = Discrete(4).n

        self._actor_network: ActorNetwork = ActorNetwork(
            output_dimensions=self._action_dimensions,
            device=self._device
        )
        self._critic_network: CriticNetwork = CriticNetwork(
            device=self._device
        )

        self._actor_network.share_memory()
        self._critic_network.share_memory()

        self._actor_network_optimizer: SharedAdam = SharedAdam(
            params=self._actor_network.parameters(),
            lr=self._learning_rate
        )
        self._critic_network_optimizer: SharedAdam = SharedAdam(
            params=self._critic_network.parameters(),
            lr=self._learning_rate
        )

        self._training_rewards: list[float] = []
        self._training_time_steps: list[int] = []
        self._training_episode_numbers: list[int] = []
        self._training_episode_rewards: list[float] = []
        self._training_episode_time_steps: list[int] = []
        self._update_numbers: list[int] = []
        self._actor_loss_history: list[float] = []
        self._critic_loss_history: list[float] = []
        self._rollout_event_count: int = 0
        self._update_log_interval: int = 250
        self._rollout_log_interval: int = 500

    @classmethod
    def create_smoke_test_agent(cls) -> "WareHouseAgentA3C":
        return cls(
            environment_class=WareHouseEnv,
            number_of_workers=1,
            worker_rollout_steps=8,
            max_global_time_steps=64,
            max_time_steps_per_episode=32,
            checkpoint_interval_updates=1_000,
            resume_from_latest_checkpoint=False,
            checkpoint_directory_name="model_weights_a3c_smoke",
            plot_directory_name="model_plots_a3c_smoke",
            base_seed=123
        )

    def train_agent(self) -> None:
        start_update_number: int = 0
        start_global_time_step: int = 0

        if self._resume_from_latest_checkpoint:
            start_update_number, start_global_time_step = self._load_checkpoint_to_resume()

        if start_global_time_step >= self._max_global_time_steps:
            self._logger.info(
                "Configured max global timesteps has already been reached by the resume checkpoint."
            )
            return

        worker_config = A3CWorkerConfig(
            gamma=self._gamma,
            entropy_coefficient=self._entropy_coefficient,
            critic_coefficient=self._critic_coefficient,
            max_grad_norm=self._max_grad_norm,
            action_dimensions=self._action_dimensions,
            worker_rollout_steps=self._worker_rollout_steps,
            max_global_time_steps=self._max_global_time_steps,
            max_time_steps_per_episode=self._max_time_steps_per_episode,
            base_seed=self._base_seed,
            environment_class=self._environment_class,
        )

        mp_context = mp.get_context("spawn")
        optimizer_lock = mp_context.Lock()
        metrics_queue = mp_context.Queue()
        global_time_step_value = mp_context.Value("i", start_global_time_step)
        global_episode_value = mp_context.Value("i", len(self._training_episode_numbers))
        global_update_value = mp_context.Value("i", start_update_number)

        progress_bar: tqdm = tqdm(
            total=self._max_global_time_steps - start_global_time_step,
            desc=f"Training Warehouse A3C Agent: {self._environment_class.__name__}"
        )

        worker_processes_list: list[mp.Process] = []

        for worker_id in range(self._number_of_workers):
            worker_process = mp_context.Process(
                target=_run_a3c_worker,
                args=(
                    worker_id,
                    worker_config,
                    self._actor_network,
                    self._critic_network,
                    self._actor_network_optimizer,
                    self._critic_network_optimizer,
                    optimizer_lock,
                    metrics_queue,
                    global_time_step_value,
                    global_episode_value,
                    global_update_value,
                )
            )
            worker_process.start()
            worker_processes_list.append(worker_process)

        try:
            while any(worker_process.is_alive() for worker_process in worker_processes_list):
                self._drain_metrics_queue(
                    metrics_queue=metrics_queue,
                    progress_bar=progress_bar
                )

            self._drain_metrics_queue(
                metrics_queue=metrics_queue,
                progress_bar=progress_bar,
                drain_all=True
            )

            failed_workers_list = [
                worker_process.exitcode
                for worker_process in worker_processes_list
                if worker_process.exitcode not in (0, None)
            ]

            if failed_workers_list:
                raise RuntimeError(f"A3C worker failure detected. Exit codes: {failed_workers_list}")

        finally:
            for worker_process in worker_processes_list:
                worker_process.join(timeout=5)
                if worker_process.is_alive():
                    worker_process.terminate()
                    worker_process.join(timeout=5)

            progress_bar.close()

        final_update_number: int = int(global_update_value.value)
        final_global_time_step: int = int(global_time_step_value.value)

        self._save_checkpoint(
            current_training_iteration=final_update_number,
            current_global_time_step=final_global_time_step
        )
        self._plot_all_training_curves()

    def get_training_history(self) -> TrainingHistory:
        sorted_time_steps, sorted_rewards = self._get_sorted_reward_history_by_time_step()
        (
            sorted_episode_numbers,
            sorted_episode_rewards,
            sorted_episode_time_steps,
        ) = self._get_sorted_reward_history_by_episode()

        return TrainingHistory(
            algorithm_name="A3C",
            environment_name=self._environment_class.__name__,
            training_time_steps=sorted_time_steps,
            training_rewards=sorted_rewards,
            training_episode_numbers=sorted_episode_numbers,
            training_episode_rewards=sorted_episode_rewards,
            training_episode_time_steps=sorted_episode_time_steps,
        )

    def _drain_metrics_queue(self, metrics_queue, progress_bar: tqdm, drain_all: bool = False) -> None:
        while True:
            try:
                timeout_seconds: float = 0.1 if drain_all else 1.0
                event_tuple = metrics_queue.get(timeout=timeout_seconds)
            except queue.Empty:
                break

            event_type_str: str = event_tuple[0]

            if event_type_str == "rollout":
                _, worker_id, rollout_length, rollout_step_metrics_list = event_tuple

                self._rollout_event_count += 1

                if (
                    self._rollout_event_count <= self._number_of_workers
                    or self._rollout_event_count % self._rollout_log_interval == 0
                ):
                    self._logger.info(
                        f"[A3C] Worker {worker_id} sent rollout with {rollout_length} step(s)."
                    )

                for time_step_int, reward_float in rollout_step_metrics_list:
                    self._training_time_steps.append(time_step_int)
                    self._training_rewards.append(reward_float)

                progress_bar.update(len(rollout_step_metrics_list))
                continue

            if event_type_str == "worker_started":
                _, worker_id, worker_seed = event_tuple
                self._logger.info(
                    f"[A3C] Worker {worker_id} started with seed {worker_seed}."
                )
                continue

            if event_type_str == "episode":
                _, episode_number, episode_reward, episode_time_step, episode_length, episode_success = event_tuple
                self._training_episode_numbers.append(int(episode_number))
                self._training_episode_rewards.append(float(episode_reward))
                self._training_episode_time_steps.append(int(episode_time_step))
                continue

            if event_type_str == "update":
                (
                    _,
                    update_number,
                    worker_id,
                    actor_loss,
                    critic_loss,
                    entropy,
                    global_step,
                    mean_episode_reward,
                    mean_episode_length,
                    success_rate,
                ) = event_tuple

                self._update_numbers.append(int(update_number))
                self._actor_loss_history.append(float(actor_loss))
                self._critic_loss_history.append(float(critic_loss))

                should_log_update_summary: bool = (
                    int(update_number) <= self._number_of_workers
                    or int(update_number) % self._update_log_interval == 0
                    or self._is_save_point(current_training_iteration=int(update_number))
                )

                if should_log_update_summary:
                    self._logger.info(
                        f"[A3C] Update={update_number} | Worker={worker_id} | "
                        f"Global Step={global_step} | Actor Loss={actor_loss:.6f} | "
                        f"Critic Loss={critic_loss:.6f} | Entropy={entropy:.6f} | "
                        f"Mean Episode Reward={mean_episode_reward:.4f} | "
                        f"Mean Episode Length={mean_episode_length:.2f} | "
                        f"Success Rate={success_rate:.2f}"
                    )
                    self._logger.info("=" * 100)

                if self._is_save_point(current_training_iteration=int(update_number)):
                    self._save_checkpoint(
                        current_training_iteration=int(update_number),
                        current_global_time_step=int(global_step)
                    )
                    if self._should_plot_at_update(int(update_number)):
                        self._plot_all_training_curves()
                continue

            if event_type_str == "error":
                _, worker_id, exception_message, traceback_string = event_tuple
                self._logger.error(f"A3C worker {worker_id} error: {exception_message}")
                self._logger.error(traceback_string)
                continue

    def _save_checkpoint(self, current_training_iteration: int, current_global_time_step: int) -> None:
        file_path: Path = self._get_file_path(current_training_iteration=current_training_iteration)

        checkpoint_dict: dict[str, int | OrderedDict | dict] = {
            "current_training_iteration": current_training_iteration,
            "current_global_time_step": current_global_time_step,
            "actor_state_dict": self._actor_network.state_dict(),
            "critic_state_dict": self._critic_network.state_dict(),
            "actor_optimizer_state_dict": self._actor_network_optimizer.state_dict(),
            "critic_optimizer_state_dict": self._critic_network_optimizer.state_dict(),
        }

        torch.save(checkpoint_dict, file_path)

        self._logger.info("\n")
        self._logger.info("=" * 100)
        self._logger.info(f"Successfully saved: {file_path}")
        self._logger.info("=" * 100)

    def _plot_rewards_by_episode(self) -> None:
        sorted_episode_numbers, sorted_episode_rewards, _ = self._get_sorted_reward_history_by_episode()
        actual_episode_number: int = sorted_episode_numbers[-1] if sorted_episode_numbers else 0
        file_path_reward_by_episode: Path = self._get_plot_file_path(
            file_path_str="rewards_by_episode",
            current_training_iteration=actual_episode_number,
            file_type_str="png"
        )
        self._model_plotting.plot_rewards_by_episode(
            file_path=file_path_reward_by_episode,
            training_episode_numbers=sorted_episode_numbers,
            training_episode_rewards=sorted_episode_rewards
        )

    def _plot_rewards_by_time_step(self) -> None:
        sorted_time_steps, sorted_rewards = self._get_sorted_reward_history_by_time_step()
        actual_time_step: int = sorted_time_steps[-1] if sorted_time_steps else 0
        file_path_rewards_by_timestep: Path = self._get_plot_file_path(
            file_path_str="rewards_by_time_step",
            current_training_iteration=actual_time_step,
            file_type_str="png"
        )
        self._model_plotting.plot_rewards_by_time_step(
            file_path=file_path_rewards_by_timestep,
            training_time_steps=sorted_time_steps,
            training_rewards=sorted_rewards
        )

    def _plot_actor_and_critic_losses_by_update(self) -> None:
        actual_update_number: int = self._update_numbers[-1] if self._update_numbers else 0
        file_path_actor_critic_loss: Path = self._get_plot_file_path(
            file_path_str="actor_critic_losses_by_update",
            current_training_iteration=actual_update_number,
            file_type_str="png"
        )
        self._model_plotting.plot_actor_and_critic_losses_by_update(
            file_path=file_path_actor_critic_loss,
            update_numbers=self._update_numbers,
            actor_loss_history=self._actor_loss_history,
            critic_loss_history=self._critic_loss_history
        )

    def _plot_all_training_curves(self) -> None:
        self._plot_rewards_by_episode()
        self._plot_rewards_by_time_step()
        self._plot_actor_and_critic_losses_by_update()

    def _get_sorted_reward_history_by_time_step(self) -> tuple[list[int], list[float]]:
        sorted_time_step_pairs_list = sorted(
            zip(self._training_time_steps, self._training_rewards),
            key=lambda pair: pair[0]
        )

        if not sorted_time_step_pairs_list:
            return [], []

        sorted_time_steps_list = [time_step for time_step, _ in sorted_time_step_pairs_list]
        sorted_rewards_list = [reward for _, reward in sorted_time_step_pairs_list]

        return sorted_time_steps_list, sorted_rewards_list

    def _get_sorted_reward_history_by_episode(self) -> tuple[list[int], list[float], list[int]]:
        sorted_episode_triplets_list = sorted(
            zip(
                self._training_episode_numbers,
                self._training_episode_rewards,
                self._training_episode_time_steps,
            ),
            key=lambda pair: pair[0]
        )

        if not sorted_episode_triplets_list:
            return [], [], []

        sorted_episode_numbers_list = [episode_number for episode_number, _, _ in sorted_episode_triplets_list]
        sorted_episode_rewards_list = [reward for _, reward, _ in sorted_episode_triplets_list]
        sorted_episode_time_steps_list = [time_step for _, _, time_step in sorted_episode_triplets_list]

        return sorted_episode_numbers_list, sorted_episode_rewards_list, sorted_episode_time_steps_list

    def _should_plot_at_update(self, current_training_iteration: int) -> bool:
        return (
            current_training_iteration > 0
            and current_training_iteration % self._plot_interval_updates == 0
        )

    def _load_checkpoint_to_resume(self) -> tuple[int, int]:
        checkpoint_directory_path: Path = Path(self._checkpoint_directory_name)

        if not checkpoint_directory_path.is_dir():
            return 0, 0

        checkpoint_paths_list: list[Path] = sorted(
            checkpoint_directory_path.glob("*.pt"),
            key=lambda path: path.stat().st_mtime
        )

        if len(checkpoint_paths_list) == 0:
            return 0, 0

        checkpoint_path_to_resume: Path

        if self._resume_checkpoint_filename is not None:
            candidate_checkpoint_path = checkpoint_directory_path / self._resume_checkpoint_filename
            if candidate_checkpoint_path.is_file():
                checkpoint_path_to_resume = candidate_checkpoint_path
            else:
                self._logger.info(
                    f"Requested resume checkpoint not found: {candidate_checkpoint_path}. "
                    f"Falling back to latest checkpoint."
                )
                checkpoint_path_to_resume = checkpoint_paths_list[-1]
        else:
            checkpoint_path_to_resume = checkpoint_paths_list[-1]

        checkpoint_dict: dict[str, int | OrderedDict | dict] = torch.load(
            checkpoint_path_to_resume,
            map_location=self._device
        )

        self._actor_network.load_state_dict(checkpoint_dict["actor_state_dict"])
        self._critic_network.load_state_dict(checkpoint_dict["critic_state_dict"])
        self._actor_network_optimizer.load_state_dict(checkpoint_dict["actor_optimizer_state_dict"])
        self._critic_network_optimizer.load_state_dict(checkpoint_dict["critic_optimizer_state_dict"])
        self._actor_network_optimizer.share_memory()
        self._critic_network_optimizer.share_memory()

        self._actor_network.to(self._device)
        self._critic_network.to(self._device)

        resumed_update_number: int = int(checkpoint_dict["current_training_iteration"])
        resumed_global_time_step: int = int(checkpoint_dict.get("current_global_time_step", 0))

        self._logger.info("=" * 100)
        self._logger.info(f"Resuming training from: {checkpoint_path_to_resume}")
        self._logger.info(f"Starting after update: {resumed_update_number}")
        self._logger.info(f"Starting after global timestep: {resumed_global_time_step}")
        self._logger.info("=" * 100)

        return resumed_update_number, resumed_global_time_step

    def _is_save_point(self, current_training_iteration: int) -> bool:
        return (
            current_training_iteration > 0
            and current_training_iteration % self._checkpoint_interval_updates == 0
        )

    def _get_file_path(self, current_training_iteration: int) -> Path:
        model_weights_directory_path: Path = Path(self._checkpoint_directory_name)
        model_weights_directory_path.mkdir(parents=True, exist_ok=True)

        self._logger.info("=" * 100)
        self._logger.info(f"Successfully created directory: {model_weights_directory_path}")
        self._logger.info("=" * 100)

        timestamp_string: str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        filename: str = f"a3c_checkpoint_step_{current_training_iteration}_{timestamp_string}.pt"

        return model_weights_directory_path / filename

    def _get_plot_file_path(self, file_path_str: str, current_training_iteration: int, file_type_str: str) -> Path:
        plotting_directory_path: Path = Path(self._plot_directory_name)
        plotting_directory_path.mkdir(parents=True, exist_ok=True)

        filename_str: str = (
            f"{file_path_str}_{current_training_iteration}_{self._timestamp_string}.{file_type_str}"
        )
        return plotting_directory_path / filename_str

    def _get_device(self):
        self._logger.info("Using Device CPU for A3C shared-memory training")
        return torch.device("cpu")
