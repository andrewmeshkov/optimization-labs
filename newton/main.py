import numpy as np
from typing import Callable, Optional
import matplotlib.pyplot as plt

from function import Hessian, Gradient, Function
from .step_strategy import StepStrategy


class NewtonOptimizer:
    def __init__(
            self,
            func: Function,
            grad: Gradient,
            hess: Hessian,
            step_strategy: StepStrategy,
            max_iterations: int = 1000,
            epsilon: float = 1e-6,
    ):
        if func.dim != grad.dim or func.dim != hess.dim:
            raise ValueError("Function, gradient and hessian must have same dimension")

        self.func = func
        self.grad = grad
        self.hess = hess
        self.step_strategy = step_strategy
        self.max_iterations = max_iterations
        self.epsilon = epsilon

    def optimize(self, make_plot: bool = False, plot_name: Optional[str] = None) -> np.ndarray:
        x = np.zeros(self.func.dim)
        trajectory = [x.copy()]

        for _ in range(self.max_iterations):
            grad_x = self.grad(x)
            if np.linalg.norm(grad_x) ** 2 < self.epsilon:
                break

            hess_x = self.hess(x)
            try:
                direction = np.linalg.solve(hess_x, grad_x)
            except np.linalg.LinAlgError:
                print("Hessian is singular, stopping optimization.")
                break

            step_size = self.step_strategy.step(x, direction, self.func, self.grad)
            x = x - step_size * direction
            trajectory.append(x.copy())

        if make_plot and len(x) == 2:
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
        plt.title(f"Newton method trajectory: {plot_name}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.grid(True)

        if plot_name:
            plt.savefig(f"{plot_name}.png")
        else:
            plt.show()

        plt.close()

class BFGSOptimizer:
    def __init__(
        self,
        func: Callable[[np.ndarray], float],
        grad: Callable[[np.ndarray], np.ndarray],
        max_iter: int = 1000,
        epsilon: float = 1e-6
    ):
        self.func = func
        self.grad = grad
        self.max_iter = max_iter
        self.epsilon = epsilon

    def optimize(self, x0: np.ndarray, make_plot: bool = False, plot_name: Optional[str] = None) -> np.ndarray:
        x = x0.copy()
        n = x.shape[0]
        I = np.eye(n)
        H = I
        trajectory = [x.copy()]

        for _ in range(self.max_iter):
            g = self.grad(x)
            if np.linalg.norm(g) < self.epsilon:
                break

            p = -H @ g
            alpha = self._line_search(x, p, g)
            x_new = x + alpha * p
            s = x_new - x
            y = self.grad(x_new) - g

            if y @ s > 1e-10:
                rho = 1.0 / (y @ s)
                V = I - rho * np.outer(s, y)
                H = V @ H @ V.T + rho * np.outer(s, s)

            x = x_new
            trajectory.append(x.copy())

        return x

    def _line_search(self, x, p, g, alpha=1.0, beta=0.5, c=1e-4):
        fx = self.func(x)
        while self.func(x + alpha * p) > fx + c * alpha * g @ p:
            alpha *= beta
        return alpha


