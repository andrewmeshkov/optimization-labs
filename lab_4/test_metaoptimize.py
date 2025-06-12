import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from function import function
from gradient_descent.step_strategy import ConstantStrategy
from linear_regression import LinearRegression
from simulated_annealing import SimulatedAnnealingOptimizer

scaler = StandardScaler()
df = pd.read_csv("lab_3/data/winequality-red.csv", sep=";")
X_raw = df.drop(columns=['quality']).values
X_raw_scaled = scaler.fit_transform(X_raw)
X = np.hstack([X_raw_scaled, np.ones((X_raw.shape[0], 1))])
y = df['quality'].values


@function(dim=1)
def batch_size_loss(x: np.ndarray) -> float:
    bs = int(np.clip(np.round(x[0]), 1, len(X)))

    model = LinearRegression(
        step_strategy=ConstantStrategy(learning_rate=0.01),
        batch_size=bs,
        max_iterations=1000,
        epsilon=1e-6,
        reg_type='elasticnet',
        alpha=0.1,
        l1_ratio=0.5,
    )

    model.fit(X, y)
    y_pred = model.predict(X)
    return float(np.mean((y - y_pred) ** 2))


def main():
    print("Test #1: batch_size")
    optimizer = SimulatedAnnealingOptimizer(
        func=batch_size_loss,
        initial_temperature=100,
        final_temperature=0.1,
        alpha=0.1,
        max_iterations=1000,
        x_0=np.array([32.0])
    )

    best_x = optimizer.optimize()
    best_bs = int(np.clip(np.round(best_x[0]), 1, len(X)))
    print(best_bs)

if __name__ == "__main__":
    main()
