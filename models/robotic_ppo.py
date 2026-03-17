from logger.logger import AppLogger


class RoboticPPO:

    def __init__(self) -> None:
        self._logger = AppLogger.get_logger(self.__class__.__name__)

    def get_proximal_policy(self, observation_dict: dict) -> int:
        return 0
