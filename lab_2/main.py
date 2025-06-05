from typing import List
import numpy as np

from newton import NewtonOptimizer, BFGSOptimizer
from newton.step_strategy import BacktrackingStepStrategy, ConstantStepStrategy
from test_functions import TEST_HESS

from scipy.optimize import minimize


def get_newton_strategies() -> List:
    return [
        ("backtracking", BacktrackingStepStrategy(alpha=1.0, beta=0.5, c=1e-4)),
        ("constant", ConstantStepStrategy(alpha=1.0)),
    ]


def main():
    print("=== Newton method (custom) ===")
    for num, func in enumerate(TEST_HESS):
        results = []
        for strategy_name, strategy in get_newton_strategies():
            solver = NewtonOptimizer(
                func=func.get_func(),
                grad=func.get_grad(),
                hess=func.get_hess(),
                step_strategy=strategy,
            )
            temp_res = solver.optimize(
                make_plot=False,
                plot_name=f"{func.get_name()}-{strategy_name}"
            )
            results.append(
                f"[{", ".join(str(el) for el in temp_res)}], iters: {solver.iterations}"
            )

        print(f"Test #{num + 1} ({func.get_name()}):")
        for (strategy_name, _), result in zip(get_newton_strategies(), results):
            print(f"\t{strategy_name}: {result}")

    print("\n=== BFGS method (custom) ===")
    for num, func in enumerate(TEST_HESS):
        x0 = np.zeros(func.get_func().dim)
        solver = BFGSOptimizer(func=func.get_func(), grad=func.get_grad())
        result = solver.optimize(x0=x0)
        print(f"Test #{num + 1} ({func.get_name()}):\n\tcustom_bfgs: {result}")

    print("\n=== scipy.optimize methods ===")
    scipy_methods = [
        ("Newton-CG", {"method": "Newton-CG", "jac": True, "hess": True}),
        ("BFGS", {"method": "BFGS", "jac": True}),
        ("L-BFGS-B", {"method": "L-BFGS-B", "jac": True}),
    ]

    for num, func in enumerate(TEST_HESS):
        print(f"Test #{num + 1} ({func.get_name()}):")
        x0 = np.zeros(func.get_func().dim)

        for method_name, options in scipy_methods:
            kwargs = {
                "fun": func.get_func(),
                "x0": x0,
                "jac": func.get_grad(),
                "method": options["method"]
            }

            if options.get("hess"):
                kwargs["hess"] = func.get_hess()

            res = minimize(**kwargs)
            print(f"\t{method_name}: {res.x}, iters: {res.nit}, success: {res.success}")


if __name__ == "__main__":
    main()
