from typing import Callable, Protocol

import numpy as np

class Function(Protocol):
    dim: int
    def __call__(self, x: np.ndarray) -> float: ...

class Gradient(Protocol):
    dim: int
    def __call__(self, x: np.ndarray) -> np.ndarray: ...

def function(dim: int):
    def decorator(func: Callable[[np.ndarray], float]):
        func.dim = dim
        return func
    return decorator

def gradient(dim: int):
    def decorator(func: Callable[[np.ndarray], np.ndarray]):
        func.dim = dim
        return func
    return decorator