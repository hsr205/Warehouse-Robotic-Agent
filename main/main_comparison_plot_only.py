from logging import Logger

from logger.logger import AppLogger
from main.main_comparison import (
    COMPARISON_SNAPSHOT_TIMESTEP_INTERVAL,
    ENVIRONMENT_CLASSES,
    load_training_histories_for_environment,
)
from utils.model_plotting import ModelPlotting


def main() -> int:
    logger: Logger = AppLogger().get_logger(__name__)
    model_plotting: ModelPlotting = ModelPlotting()

    try:
        for environment_class in ENVIRONMENT_CLASSES:
            training_histories_dict = load_training_histories_for_environment(environment_class=environment_class)

            if len(training_histories_dict) == 0:
                logger.info(f"No saved histories found for environment: {environment_class.__name__}")
                continue

            environment_obj = environment_class(render_mode=None)

            try:
                model_plotting.create_comparison_plots_for_environment(
                    environment_obj=environment_obj,
                    training_histories_dict=training_histories_dict,
                )
                model_plotting.create_timestep_snapshot_plots_for_environment(
                    environment_obj=environment_obj,
                    training_histories_dict=training_histories_dict,
                    time_step_interval=COMPARISON_SNAPSHOT_TIMESTEP_INTERVAL,
                )
                model_plotting.create_episode_snapshot_plots_for_environment(
                    environment_obj=environment_obj,
                    training_histories_dict=training_histories_dict,
                    time_step_interval=COMPARISON_SNAPSHOT_TIMESTEP_INTERVAL,
                )
            finally:
                environment_obj.close()

        return 0

    except Exception as exception:
        logger.error(f"Exception Thrown: {exception}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
