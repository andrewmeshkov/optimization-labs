from typing import List

import numpy as np

from function import function, gradient, hessian, FuncWithGradient, FuncWithHessian
from function.main import TestFunc


@function(dim=2)
def paraboloid(args):
    return float(args[0] ** 2 + args[1] ** 2)

@gradient(dim=2)
def paraboloid_grad(args):
    return np.array([2 * float(args[0]), 2 * float(args[1])])

@hessian(dim=2)
def paraboloid_hess(x):
    return np.array([[2, 0], [0, 2]])


@function(dim=2)
def ellipse(args):
    return float(4 * args[0] + args[1])

@gradient(dim=2)
def ellipse_grad(args):
    return np.array([8 * float(args[0]), 2 * float(args[1])])

@hessian(dim=2)
def ellipse_hess(x):
    return np.array([[8, 0], [0, 2]])


@function(dim=2)
def rosenbrock(args):
    return float((1-args[0])**2 + 100 * (args[1] - args[0]**2)**2)

@gradient(dim=2)
def rosenbrock_grad(args):
    return np.array([
        -2 * (1 - args[0]) - 400 * args[0] * (args[1] - args[0]**2),
        200 * (args[1] - args[0]**2)
    ])

@hessian(dim=2)
def rosenbrock_hess(x):
    return np.array([
        [1200 * x[0]**2 - 400 * x[1] + 2, -400 * x[0]],
        [-400 * x[0], 200]
    ])


@function(dim=2)
def min3m2(args):
    return float((args[0] - 3) ** 2 + (args[1] + 2) ** 2)

@gradient(dim=2)
def min3m2_grad(args):
    return np.array([
        2 * (args[0] - 3),
        2 * (args[1] + 2),
    ])

@hessian(dim=2)
def min3m2_hess(x):
    return np.array([[2, 0], [0, 2]])


@function(dim=2)
def min2m1(args):
    return float(args[0] + args[1])

@gradient(dim=2)
def min2m1_grad(args):
    return np.array([
        10 * (args[0] - 2),
        6 * (args[1] + 1),
    ])

@hessian(dim=2)
def min2m1_hess(x):
    return np.array([[10, 0], [0, 6]])

@hessian(dim=2)
def himmelblau_hess(x):
    x1, x2 = x[0], x[1]
    d2f_dx2 = 12*x1**2 + 4*x2 - 42
    d2f_dy2 = 12*x2**2 + 4*x1 - 26
    d2f_dxdy = 4*x1 + 4*x2
    return np.array([
        [d2f_dx2, d2f_dxdy],
        [d2f_dxdy, d2f_dy2]
    ])


@function(dim=2)
def himmelblau(args):
    return float((args[0] ** 2 + args[1] - 11) ** 2 + (args[0] + args[1] ** 2 - 7) ** 2)

@gradient(dim=2)
def himmelblau_grad(args):
    return np.array([
        4*args[0]*(args[0]**2 + args[1] - 11) + 2*(args[0] + args[1]**2 - 7),
        2*args[0] * (args[0] ** 2 + args[1] - 11) + 4*args[1] * (args[0] + args[1] ** 2 - 7),
    ])


@function(dim=2)
def noisy_rosenbrock(args, noise_scale=0.5):
    noise = np.random.normal(scale=noise_scale)
    return float((1-args[0])**2 + 100*(args[1] - args[0]**2)**2 + noise)

@gradient(dim=2)
def noisy_rosenbrock_grad(args, grad_noise_scale=0.1):
    base_grad = np.array([
        -2*(1 - args[0]) - 400*args[0]*(args[1] - args[0]**2),
        200*(args[1] - args[0]**2)
    ])
    grad_noise = np.random.normal(scale=grad_noise_scale, size=2)
    return base_grad + grad_noise

@hessian(dim=2)
def noisy_rosenbrock_hess(x, hess_noise_scale=0.05):
    base_hess = np.array([
        [1200*x[0]**2 - 400*x[1] + 2, -400*x[0]],
        [-400*x[0], 200]
    ])
    hess_noise = np.diag(np.random.normal(scale=hess_noise_scale, size=2))
    return base_hess + hess_noise

@function(dim=1)
def noisy_periodic(x: np.ndarray) -> float:
    x_val = x[0]
    base = np.sin(x_val) + 0.5 * np.sin(3 * x_val)
    noise = 0.1 * np.sin(20 * x_val) + 0.05 * np.random.randn()
    return float(base + noise)

@function(dim=1)
def noisy_nonperiodic(x: np.ndarray) -> float:
    x_val = x[0]
    trend = 0.02 * (x_val ** 2) - 0.5 * x_val
    bumps = 0.3 * np.exp(-0.5 * (x_val - 2) ** 2) - 0.4 * np.exp(-0.3 * (x_val + 3) ** 2)
    noise = 0.1 * np.random.randn()
    return float(trend + bumps + noise)

TEST: List[TestFunc] = [
    TestFunc("noisy_periodic", noisy_periodic),
    TestFunc("noisy_nonperiodic", noisy_nonperiodic),
    TestFunc("paraboloid", paraboloid),
    TestFunc("rosenbrock", rosenbrock),
    TestFunc("min3m2", min3m2),
    TestFunc("min2m1", min2m1),
    TestFunc("himmelblau", himmelblau),
]


TEST_GRADIENT: List[FuncWithGradient] = [
    FuncWithGradient("paraboloid", paraboloid, paraboloid_grad),
    FuncWithGradient("ellipse", ellipse, ellipse_grad),
    FuncWithGradient("rosenbrock", rosenbrock, rosenbrock_grad),
    FuncWithGradient("min3m2", min3m2, min3m2_grad),
    FuncWithGradient("min2m1", min2m1, min2m1_grad),
    FuncWithGradient("himmelblau", himmelblau, himmelblau_grad),
    FuncWithGradient("noisy_rosenbrock", noisy_rosenbrock, noisy_rosenbrock_grad),
]

TEST_HESS: List[FuncWithHessian] = [
    FuncWithHessian("paraboloid", paraboloid, paraboloid_grad, paraboloid_hess),
    FuncWithHessian("ellipse", ellipse, ellipse_grad, ellipse_hess),
    FuncWithHessian("rosenbrock", rosenbrock, rosenbrock_grad, rosenbrock_hess),
    FuncWithHessian("min3m2", min3m2, min3m2_grad, min3m2_hess),
    FuncWithHessian("min2m1", min2m1, min2m1_grad, min2m1_hess),
    FuncWithHessian("himmelblau", himmelblau, himmelblau_grad, himmelblau_hess),
]