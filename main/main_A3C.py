import torch.multiprocessing as mp
from logging import Logger

from logger.logger import AppLogger
from models.warehouse_agent_a3c import WareHouseAgentA3C


def main() -> int:
    logger: Logger = AppLogger().get_logger(__name__)

    warehouse_agent_a3c = WareHouseAgentA3C(
        number_of_workers=2,
        worker_rollout_steps=32,
        max_global_time_steps=1_000_000,
        max_time_steps_per_episode=500,
        checkpoint_interval_updates=500,
        resume_from_latest_checkpoint=True,
        checkpoint_directory_name="model_weights_a3c",
        plot_directory_name="model_plots_a3c",
        base_seed=42
    )

    try:
        logger.info(
            "Starting baseline A3C training on warehouse_env_2 with "
            "2 workers, rollout_steps=32, max_global_timesteps=1000000."
        )
        warehouse_agent_a3c.train_agent()
        return 0

    except Exception as e:
        logger.error(f"Exception Thrown: {e}")

    return 1


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
