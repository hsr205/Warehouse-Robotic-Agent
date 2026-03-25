from dataclasses import dataclass


@dataclass
class StepDataObj:
    observation_dict: dict
    action_int: int
    action_str: str
    reward_float: float
    is_done: bool
    info_dict: dict
