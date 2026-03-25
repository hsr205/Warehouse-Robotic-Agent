import math


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
