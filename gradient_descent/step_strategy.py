from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from gradient_descent.function import Function


class StepStrategy(ABC):
    @abstractmethod
    def step(
            self,
            prev_x: np.ndarray,
            function: float,
            gradient: np.ndarray,
    ) -> float:
        pass

class ConstantStepStrategy(StepStrategy):
    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate

    def step(self, prev_x: np.ndarray, function: float, gradient: np.ndarray) -> float:
        return self.learning_rate

class PiecewiseConstantStepStrategy(StepStrategy):
    def __init__(
            self,
            initial_value: float,
            reduction_factor: float,
            patience: int,
            tol: float,
    ):
        self.current_step = initial_value
        self.reduction_factor = reduction_factor
        self.patience = patience
        self.tol = tol
        self.prev_grad_norm = np.inf
        self.no_improvement_count = 0

    def step(self, prev_x: np.ndarray, function: float, gradient: np.ndarray) -> float:
        grad_norm = np.linalg.norm(gradient)

        if grad_norm < self.prev_grad_norm - self.tol:
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1

        if self.no_improvement_count >= self.patience:
            self.current_step *= self.reduction_factor
            self.no_improvement_count = 0

        self.prev_grad_norm = np.linalg.norm(gradient)
        return self.current_step

class SteepestGradientStrategy(StepStrategy):
    def __init__(self, func: Function, eps: float = 1e-5):
        self.func = func
        self.eps = eps
        self.GOLDEN_RATIO = (1 + np.sqrt(5)) / 2

    def step(self, prev_x: np.ndarray, function: float, gradient: np.ndarray) -> float:
        l, r = 0.0, 5.0

        while abs(r - l) > self.eps:
            x1 = r - (r - l) / self.GOLDEN_RATIO
            x2 = l + (r - l) / self.GOLDEN_RATIO

            point1 = prev_x - x1 * gradient
            point2 = prev_x - x2 * gradient

            y1 = self.func(point1)
            y2 = self.func(point2)

            if y1 >= y2:
                l = x1
            else:
                r = x2

        return (l + r) / 2