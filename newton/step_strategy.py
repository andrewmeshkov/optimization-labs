from abc import ABC, abstractmethod
from typing import Callable

import numpy as np

class StepStrategy(ABC):
    @abstractmethod
    def step(self, x: np.ndarray, direction: np.ndarray, func: Callable[[np.ndarray], float], grad: Callable[[np.ndarray], np.ndarray]) -> float:
        pass


class BacktrackingStepStrategy(StepStrategy):
    def __init__(self, alpha: float = 1.0, beta: float = 0.5, c: float = 1e-4):
        self.alpha = alpha
        self.beta = beta
        self.c = c

    def step(self, x: np.ndarray, direction: np.ndarray, func: Callable[[np.ndarray], float], grad: Callable[[np.ndarray], np.ndarray]) -> float:
        alpha = self.alpha
        fx = func(x)
        grad_dot_dir = grad(x).dot(direction)

        while func(x - alpha * direction) > fx - self.c * alpha * grad_dot_dir:
            alpha *= self.beta

        return alpha


class ConstantStepStrategy(StepStrategy):
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def step(self, x: np.ndarray, direction: np.ndarray, func: Callable[[np.ndarray], float],
             grad: Callable[[np.ndarray], np.ndarray]) -> float:
        return self.alpha
