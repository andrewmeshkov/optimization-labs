import numpy as np

from function import function, gradient, hessian, FuncWithGradient, FuncWithHessian


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



TEST_GRADIENT = [
    FuncWithGradient("paraboloid", paraboloid, paraboloid_grad),
    FuncWithGradient("ellipse", ellipse, ellipse_grad),
    FuncWithGradient("rosenbrock", rosenbrock, rosenbrock_grad),
    FuncWithGradient("min3m2", min3m2, min3m2_grad),
    FuncWithGradient("min2m1", min2m1, min2m1_grad),
    FuncWithGradient("himmelblau", himmelblau, himmelblau_grad),
]

TEST_HESS = [
    FuncWithHessian("paraboloid", paraboloid, paraboloid_grad, paraboloid_hess),
    FuncWithHessian("ellipse", ellipse, ellipse_grad, ellipse_hess),
    FuncWithHessian("rosenbrock", rosenbrock, rosenbrock_grad, rosenbrock_hess),
    FuncWithHessian("min3m2", min3m2, min3m2_grad, min3m2_hess),
    FuncWithHessian("min2m1", min2m1, min2m1_grad, min2m1_hess),
    FuncWithHessian("himmelblau", himmelblau, himmelblau_grad, himmelblau_hess),
]