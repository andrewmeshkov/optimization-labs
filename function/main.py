from typing import Callable, Protocol

import numpy as np

class Function(Protocol):
    dim: int
    def __call__(self, x: np.ndarray) -> float: ...

class Gradient(Protocol):
    dim: int
    def __call__(self, x: np.ndarray) -> np.ndarray: ...

class Hessian(Protocol):
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

def hessian(dim: int):
    def decorator(func: Callable[[np.ndarray], np.ndarray]):
        func.dim = dim
        return func
    return decorator


class FuncWithGradient:
    def __init__(self, name: str, func: Function, grad: Gradient):
        self.func = func
        self.grad = grad
        self.name = name

    def get(self) -> (Function, Gradient):
        return self.func, self.grad


    def get_func(self) -> Function:
        return self.func

    def get_grad(self) -> Gradient:
        return self.grad

    def get_name(self) -> str:
        return self.name


class FuncWithHessian(FuncWithGradient):
    def __init__(self, name: str, func: Function, grad: Gradient, hess: Hessian):
        self.hess = hess
        super().__init__(name, func, grad)

    def get(self) -> (Function, Gradient, Hessian):
        return self.func, self.grad, self.hess, self.name

    def get_hess(self) -> Hessian:
        return self.hess

