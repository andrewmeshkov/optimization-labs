from typing import List

from gradient_descent import GradientDescentOptimizer
from gradient_descent.step_strategy import StepStrategy, ConstantStepStrategy, PiecewiseConstantStepStrategy, \
    SteepestGradientStrategy
from lab_1.func_with_gradient import FuncWithGradient
from test_functions import TEST


def get_strategies(func: FuncWithGradient) -> List[StepStrategy]:
    return [
        ConstantStepStrategy(0.1),
        PiecewiseConstantStepStrategy(0.05, 0.5, 5, 1e-5),
        SteepestGradientStrategy(func.get_func())
    ]


for num, func in enumerate(TEST):
    results = []
    for strategy in get_strategies(func):
        solver = GradientDescentOptimizer(func.get_func(), func.get_grad(), strategy)
        results.append(solver.optimize(make_plot=True, plot_name=f"{func.get_name()}-{strategy.__name__}"))

    print(f"Test #{num+1} ({func.get_name()}):", *results, sep='\n\t')

