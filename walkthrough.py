import numpy as np
# Assuming your files are organized in a local directory module structure:
from linalg_regressor.engine import ProjectionRegressor
from linalg_regressor.diagnostics import GeometricDiagnostics
from linalg_regressor.robust import RobustProjectionRegressor



if __name__ == "__main__":
    print("=== PART 1: VERIFYING PROFESSOR STRANG'S LECTURE EXAMPLE ===")
    # Strang's exact points from the lecture: t=[1, 2, 3], heights b=[1, 2, 2]
    t = np.array([[1.0], [2.0], [3.0]])
    b = np.array([1.0, 2.0, 2.0])

    model = ProjectionRegressor()
    model.fit(t, b)

    print(f"Calculated Intercept (C): {model.beta[0][0]:.4f} (Expected: 0.6667 / 2/3)")
    print(f"Calculated Slope (D):     {model.beta[1][0]:.4f} (Expected: 0.5000 / 1/2)")
    print(f"Error Vector e:           {model.e.flatten()}")

    # Run geometric proofs
    diag = GeometricDiagnostics.verify_orthogonality(model)
    print(f"Is error vector orthogonal to column space? {diag['is_orthogonal']}")
    print(f"Max deviation from zero: {diag['max_deviation']}\n")


    print("=== PART 2: BENCHMARKING AGAINST OUTLIERS ===")
    # Generate clean linear data
    np.random.seed(42)
    X_clean = np.linspace(1, 10, 20).reshape(-1, 1)
    y_clean = 2.5 * X_clean.flatten() + 1.0 + np.random.normal(0, 0.5, 20)

    # Introduce a massive outlier (The outlier problem at timestamp 00:15:28)
    y_corrupted = y_clean.copy()
    y_corrupted[5] = 85.0 # Normal value should be around ~13.5

    # Fit standard model
    std_model = ProjectionRegressor()
    std_model.fit(X_clean, y_corrupted)

    # Fit robust model
    robust_model = RobustProjectionRegressor(max_trials=200, residual_threshold=2.0)
    robust_model.fit(X_clean, y_corrupted)

    print(f"Standard Model slope with outlier: {std_model.beta[1][0]:.4f} (Heavily skewed!)")
    print(f"Robust Model slope with outlier:   {robust_model.best_model.beta[1][0]:.4f} (Maintained accuracy!)")