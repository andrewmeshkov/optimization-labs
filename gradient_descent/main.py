from typing import Callable, Optional
import numpy as np
import matplotlib.pyplot as plt

from .step_strategy import StepStrategy
from function import Function, Gradient


class GradientDescentOptimizer:
    def __init__(
            self,
            func: Function,
            grad: Gradient,
            step_strategy: StepStrategy,
            max_iterations: int = 10000,
            epsilon: float = 1e-6,
            x_0: Optional[np.ndarray] = None
    ):
        if func.dim != grad.dim:
            raise ValueError("Function and gradient must have same dimension")

        self.func = func
        self.grad = grad
        self.step_strategy = step_strategy
        self.max_iterations = max_iterations
        self.epsilon = epsilon
        self.x_0 = x_0 if x_0 is not None else np.zeros(func.dim)

    def set_step_strategy(self, step_strategy: StepStrategy):
        self.step_strategy = step_strategy

    def set_function(self, func: Function, gradient: Callable[[np.ndarray], float]):
        self.func = func
        self.grad = gradient

    def optimize(self, make_plot: bool = False, plot_name: Optional[str] = None) -> np.ndarray:
        x = self.x_0.copy()
        trajectory = [x.copy()]

        for i in range(self.max_iterations):
            grad_x = self.grad(x)
            if np.linalg.norm(grad_x) ** 2 < self.epsilon:
                break
            step_size = self.step_strategy.step(x, self.func(x), grad_x)
            x = x - step_size * grad_x
            trajectory.append(x.copy())

        if make_plot and self.func.dim == 2:
            self._plot_trajectory(np.array(trajectory), plot_name)

        return x

    def _plot_trajectory(self, trajectory: np.ndarray, plot_name: Optional[str]):
        x_vals = trajectory[:, 0]
        y_vals = trajectory[:, 1]

        x = np.linspace(min(x_vals) - 1, max(x_vals) + 1, 400)
        y = np.linspace(min(y_vals) - 1, max(y_vals) + 1, 400)
        X, Y = np.meshgrid(x, y)
        Z = np.array([[self.func(np.array([xi, yi])) for xi in x] for yi in y])

        plt.figure(figsize=(6, 5))
        plt.contour(X, Y, Z, levels=30)
        plt.plot(x_vals, y_vals, marker='o', markersize=3, linewidth=1.5, label="trajectory")
        plt.scatter(x_vals[0], y_vals[0], color='green', label='start')
        plt.scatter(x_vals[-1], y_vals[-1], color='red', label='finish')
        plt.title(f"Gradient descent trajectory: {plot_name}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.grid(True)

        if plot_name:
            plt.savefig(f"{plot_name}.png")
        else:
            plt.show()

        plt.close()
