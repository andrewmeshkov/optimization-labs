from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

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