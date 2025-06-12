import numpy as np
from typing import Optional, List
from function import Function


class SimulatedAnnealingOptimizer:
    def __init__(
        self,
        func: Function,
        initial_temperature: float = 100.0,
        final_temperature: float = 1e-3,
        alpha: float = 0.9,
        max_iterations: int = 1000,
        x_0: Optional[np.ndarray] = None
    ):
        self.func = func
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.alpha = alpha
        self.max_iterations = max_iterations
        self.iterations = 0
        self.x_0 = x_0 if x_0 is not None else np.zeros(func.dim)
        self.trajectory: List[np.ndarray] = []

    def optimize(self) -> np.ndarray:
        self.trajectory = []
        x = self.x_0.copy()
        current_value = self.func(x)
        temperature = self.initial_temperature
        self.iterations = 0
        self.trajectory.append(x.copy())

        for i in range(self.max_iterations):
            if temperature < self.final_temperature:
                break

            candidate = x + np.random.normal(scale=temperature, size=x.shape)
            candidate_value = self.func(candidate)

            delta = candidate_value - current_value
            if delta < 0 or np.random.rand() < np.exp(-delta / temperature):
                x = candidate
                current_value = candidate_value
                self.trajectory.append(x.copy())

            temperature = self.initial_temperature / (1 + self.alpha * i)
            self.iterations += 1

        return x


