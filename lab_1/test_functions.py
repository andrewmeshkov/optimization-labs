import numpy as np

from gradient_descent.function import function, gradient
from lab_1.func_with_gradient import FuncWithGradient


@function(dim=2)
def paraboloid(args):
    return float(args[0] ** 2 + args[1] ** 2)

@gradient(dim=2)
def paraboloid_grad(args):
    return np.array([2 * float(args[0]), 2 * float(args[1])])


@function(dim=2)
def ellipse(args):
    return float(4 * args[0] + args[1])

@gradient(dim=2)
def ellipse_grad(args):
    return np.array([8 * float(args[0]), 2 * float(args[1])])



@function(dim=2)
def rosenbrock(args):
    return float((1-args[0])**2 + 100 * (args[1] - args[0]**2)**2)

@gradient(dim=2)
def rosenbrock_grad(args):
    return np.array([
        -2 * (1 - args[0]) - 400 * args[0] * (args[1] - args[0]**2),
        200 * (args[1] - args[0]**2)
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


@function(dim=2)
def min2m1(args):
    return float(args[0] + args[1])

@gradient(dim=2)
def min2m1_grad(args):
    return np.array([
        10 * (args[0] - 2),
        6 * (args[1] + 1),
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



TEST = [
    FuncWithGradient("paraboloid", paraboloid, paraboloid_grad),
    FuncWithGradient("ellipse", ellipse, ellipse_grad),
    FuncWithGradient("rosenbrock", rosenbrock, rosenbrock_grad),
    FuncWithGradient("min3m2", min3m2, min3m2_grad),
    FuncWithGradient("min2m1", min2m1, min2m1_grad),
    FuncWithGradient("himmelblau", himmelblau, himmelblau_grad),
]
