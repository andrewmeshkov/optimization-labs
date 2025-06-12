from typing import List

import numpy as np

from function import Function
from simulated_annealing import SimulatedAnnealingOptimizer
from test_functions import TEST

import matplotlib.pyplot as plt


class HyperParamsKit:
    def __init__(self, alpha, initial_temperature, final_temperature, max_iterations=100000):
        self.alpha = alpha
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.max_iterations = max_iterations

TEST_PARAMS: List[HyperParamsKit] = [
    HyperParamsKit(0.5, 1000, 1e-3),
    HyperParamsKit(0.9, 1000, 1e-3),
    HyperParamsKit(0.99, 1000, 1e-3),
    HyperParamsKit(0.5, 100, 1e-3),
    HyperParamsKit(0.9, 100, 1e-3),
    HyperParamsKit(0.99, 100, 1e-3),
    HyperParamsKit(0.5, 1000, 1e-5),
    HyperParamsKit(0.9, 1000, 1e-5),
    HyperParamsKit(0.99, 1000, 1e-5),
    HyperParamsKit(0.5, 100, 1e-5),
    HyperParamsKit(0.9, 100, 1e-5),
    HyperParamsKit(0.99, 100, 1e-5),
]


def save_plot(func: Function, filename: str, x_range=(-50, 50), num_points=100000):
    if func.dim != 1:
        raise ValueError("Function must be 1-dimensional to plot.")

    x_vals = np.linspace(x_range[0], x_range[1], num_points)
    y_vals = [func(np.array([x])) for x in x_vals]

    plt.figure(figsize=(8, 5))
    plt.plot(x_vals, y_vals, color='black', linewidth=1.5)
    plt.title("")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def main():
    for num, func in enumerate(TEST):
        results = []

        if func.get_name() == "noisy_nonperiodic":
            save_plot(func=func.get_func(), filename=f"lab_4/report/{func.get_name()}.png", x_range=(-5, 25), num_points=1000)
        if func.get_name() == "noisy_periodic":
            save_plot(func=func.get_func(), filename=f"lab_4/report/{func.get_name()}.png", x_range=(-10, 10), num_points=500)
        for params in TEST_PARAMS:
            solver = SimulatedAnnealingOptimizer(
                func=func.get_func(),
                initial_temperature=params.initial_temperature,
                final_temperature=params.final_temperature,
                alpha=params.alpha,
                max_iterations=params.max_iterations,
            )
            temp_res = solver.optimize()

            results.append((
                f"[{", ".join(str(el) for el in temp_res)}]",
                params,
                func.get_func()(temp_res),
            ))



        print(f"Test #{num + 1} ({func.get_name()}):")
        for value, params, func_value in results:
            print(f"start t: {params.initial_temperature}, final t: {params.final_temperature}, alpha: {params.alpha}, value: {value}, func_value: {func_value}")

if __name__ == "__main__":
    main()