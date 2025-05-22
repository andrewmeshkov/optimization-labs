from typing import List
import numpy as np

from gradient_descent import GradientDescentOptimizer
from gradient_descent.step_strategy import StepStrategy, ConstantStrategy, PiecewiseConstantsStrategy, \
    SteepestStrategyGR, SteepestStrategyDichotomy
from function import FuncWithGradient
from test_functions import TEST_GRADIENT

from scipy.optimize import minimize


def get_gd_strategies(func: FuncWithGradient) -> List[tuple[str, StepStrategy]]:
    return [
        ("constant", ConstantStrategy(0.1)),
        ("piecewise", PiecewiseConstantsStrategy(0.05, 0.5, 5, 1e-5)),
        ("steepest_gr", SteepestStrategyGR(func.get_func())),
        ("steepest_dichotomy", SteepestStrategyDichotomy(func.get_func())),
    ]


def main():
    print("=== Gradient Descent (custom) ===")
    for num, func in enumerate(TEST_GRADIENT):
        results = []
        for strategy_name, strategy in get_gd_strategies(func):
            solver = GradientDescentOptimizer(
                func=func.get_func(),
                grad=func.get_grad(),
                step_strategy=strategy,
            )

            results.append(
                solver.optimize(
                    make_plot=False,
                    plot_name=f"{func.get_name()}-{strategy_name}"
                )
            )

        print(f"Test #{num + 1} ({func.get_name()}):")
        for (strategy_name, _), result in zip(get_gd_strategies(func), results):
            print(f"\t{strategy_name}: {result}")


    print("\n=== scipy.optimize methods (for comparison) ===")
    scipy_methods = [
        ("CG", "CG"),
        ("BFGS", "BFGS"),
    ]

    for num, func in enumerate(TEST_GRADIENT):
        print(f"Test #{num + 1} ({func.get_name()}):")
        x0 = np.zeros(func.get_func().dim)
        for method_name, method in scipy_methods:
            res = minimize(
                fun=func.get_func(),
                x0=x0,
                jac=func.get_grad(),
                method=method
            )
            print(f"\t{method_name}: {res.x}, iters: {res.nit}, success: {res.success}")


if __name__ == "__main__":
    main()
