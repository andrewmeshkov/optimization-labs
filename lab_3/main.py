import time
import tracemalloc
import numpy as np
import pandas as pd

from linear_regression import LinearRegression
from gradient_descent.step_strategy import (
    ConstantStrategy,
    MomentumStrategy,
    PiecewiseConstantsStrategy,
)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df = pd.read_csv("lab_3/data/winequality-red.csv", sep=";")
X_raw = df.drop(columns=['quality']).values
X_raw_scaled = scaler.fit_transform(X_raw)
X = np.hstack([X_raw_scaled, np.ones((X_raw.shape[0], 1))])
y = df['quality'].values

def run_test(test_num, strategy_name, strategy, batch_size, reg_type,
             alpha=0.0, l1_ratio=0.5, max_iter=1000):

    print(f"Test #{test_num} [{strategy_name}] (batch size={batch_size}, regularization={reg_type})")

    model = LinearRegression(
        step_strategy=strategy,
        batch_size=batch_size,
        max_iterations=max_iter,
        epsilon=1e-6,
        reg_type=reg_type,
        alpha=alpha,
        l1_ratio=l1_ratio
    )

    tracemalloc.start()
    start_time = time.perf_counter()

    model.fit(X, y)

    end_time = time.perf_counter()
    mem_used = tracemalloc.get_traced_memory()[1] / 1024 / 1024
    tracemalloc.stop()

    y_pred = model.predict(X)
    mse = np.mean((y - y_pred) ** 2)
    ops = model.total_operations

    print(f"  MSE: {mse:.4f}")
    print(f"  Time: {end_time - start_time:.4f} s")
    print(f"  Memory peak: {mem_used:.2f} MB")
    print(f"  Total ops: {ops:,}")
    print()

batch_sizes = [1, 32, None]
regularizations = ['none', 'l2', 'l1', 'elasticnet']

test_counter = 1

for bs in batch_sizes:
    for reg in regularizations:
        strategy = ConstantStrategy(learning_rate=0.01)
        run_test(test_counter, "ConstantStrategy", strategy, bs, reg, alpha=0.1, l1_ratio=0.5)
        test_counter += 1

for bs in batch_sizes:
    for reg in regularizations:
        strategy = MomentumStrategy(learning_rate=0.01, beta=0.9)
        run_test(test_counter, "MomentumStrategy", strategy, bs, reg, alpha=0.1, l1_ratio=0.5)
        test_counter += 1


for bs in batch_sizes:
    for reg in regularizations:
        strategy = PiecewiseConstantsStrategy(
            initial_value=0.05,
            reduction_factor=0.5,
            patience=10,
            tol=1e-4
        )
        run_test(test_counter, "PiecewiseConstantsStrategy", strategy, bs, reg, alpha=0.1, l1_ratio=0.5)
        test_counter += 1
