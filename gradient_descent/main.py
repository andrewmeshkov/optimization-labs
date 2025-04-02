from typing import Callable
import numpy as np

from .step_strategy import StepStrategy
from .function import Function, Gradient


class GradientDescentOptimizer:
    def __init__(
            self,
            func: Function,
            grad: Gradient,
            step_strategy: StepStrategy,
            max_iterations: int = 10000,
            epsilon: float = 1e-6,
    ):
        if func.dim != grad.dim:
          raise ValueError("Function and gradient must have same dimension")

        self.func = func
        self.grad = grad
        self.step_strategy = step_strategy
        self.max_iterations = max_iterations
        self.epsilon = epsilon

    def set_step_strategy(self, step_strategy: StepStrategy):
        self.step_strategy = step_strategy

    def set_function(self, func: Callable[[np.ndarray], float], gradient: Callable[[np.ndarray], float]):
        self.func = func
        self.grad = gradient

    def optimize(self) -> np.ndarray:
        x = np.zeros(self.func.dim)
        for i in range(self.max_iterations):
            grad_x = self.grad(x)
            if np.linalg.norm(grad_x) ** 2 < self.epsilon:
                break
            x = x - self.step_strategy.step(x, self.func(x), grad_x) * grad_x
        return x