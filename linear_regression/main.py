import numpy as np
from typing import Optional, Literal

from gradient_descent.step_strategy import StepStrategy
from function import function, gradient, Function, Gradient


def make_regularized_mse(
    x: np.ndarray,
    y: np.ndarray,
    reg_type: Literal['none', 'l1', 'l2', 'elasticnet'] = 'none',
    alpha: float = 0.0,
    l1_ratio: float = 0.5,
) -> tuple[Function, Gradient]:
    dim = x.shape[1]

    @function(dim)
    def mse(w: np.ndarray) -> float:
        residual = x @ w - y
        loss = np.mean(residual ** 2)

        if reg_type == 'l2':
            reg = alpha * np.sum(w ** 2)
        elif reg_type == 'l1':
            reg = alpha * np.sum(np.abs(w))
        elif reg_type == 'elasticnet':
            reg = alpha * (l1_ratio * np.sum(np.abs(w)) + (1 - l1_ratio) * np.sum(w ** 2))
        else:
            reg = 0

        return loss + reg

    @gradient(dim)
    def grad(w: np.ndarray) -> np.ndarray:
        residual = x @ w - y
        g = 2 * x.T @ residual / len(y)

        if reg_type == 'l2':
            g += 2 * alpha * w
        elif reg_type == 'l1':
            g += alpha * np.sign(w)
        elif reg_type == 'elasticnet':
            g += alpha * (l1_ratio * np.sign(w) + 2 * (1 - l1_ratio) * w)

        return g

    return mse, grad


class LinearRegression:
    def __init__(
        self,
        step_strategy: StepStrategy,
        batch_size: Optional[int] = None,
        max_iterations: int = 1000,
        epsilon: float = 1e-6,
        reg_type: Literal['none', 'l1', 'l2', 'elasticnet'] = 'none',
        alpha: float = 0.0,
        l1_ratio: float = 0.5,
        x_0: Optional[np.ndarray] = None,
    ):
        self.step_strategy = step_strategy
        self.batch_size = batch_size
        self.max_iterations = max_iterations
        self.epsilon = epsilon
        self.reg_type = reg_type
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.x_0 = x_0
        self.weights: Optional[np.ndarray] = None
        self.total_operations = 0

    def _generate_batches(self, x, y):
        if self.batch_size is None:
            yield x, y
        else:
            indices = np.random.permutation(len(x))
            for i in range(0, len(x), self.batch_size):
                idx = indices[i:i + self.batch_size]
                yield x[idx], y[idx]

    def fit(self, x: np.ndarray, y: np.ndarray):
        n_features = x.shape[1]
        w = self.x_0.copy() if self.x_0 is not None else np.zeros(n_features)
        for _ in range(self.max_iterations):
            stop = False
            for x_batch, y_batch in self._generate_batches(x, y):
                self.total_operations += 4 * x_batch.shape[0] * x_batch.shape[1]
                func, grad = make_regularized_mse(
                    x_batch, y_batch,
                    self.reg_type, self.alpha, self.l1_ratio
                )

                grad_w = grad(w)
                if np.linalg.norm(grad_w) < self.epsilon:
                    stop = True
                    break

                step = self.step_strategy.step(w, func(w), grad_w)
                if hasattr(self.step_strategy, 'get_update_direction'):
                    direction = self.step_strategy.get_update_direction(grad_w)
                    w -= step * direction
                else:
                    w -= step * grad_w

            if stop:
                break

        self.weights = w

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise ValueError("Model is not trained yet")
        return x @ self.weights
