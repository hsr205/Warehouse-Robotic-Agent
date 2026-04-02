class Constants:
    LOGGER_COLOR_RESET: str = "\033[0m"
    LOGGER_COLOR_WHITE: str = "\033[60m"
    LOGGER_COLOR_ORANGE: str = "\033[33m"
    LOGGER_COLOR_DARK_RED: str = "\033[31m"

    ACTION_SPACE_MAPPING_DICT: dict[int, str] = {
        0: "LEFT",
        1: "RIGHT",
        2: "FORWARD",
        3: "PICKUP_OBJECT",
        4: "DROP_OBJECT",
        5: "TOGGLE",
        6: "DONE_COMPLETING_TASK",
    }

    AGENT_DIRECTION_MAPPING_DICT: dict[int, str] = {
        0: "RIGHT",
        1: "DOWN",
        2: "LEFT",
        3: "UP",
    }
