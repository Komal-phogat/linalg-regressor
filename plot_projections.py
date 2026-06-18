import numpy as np
import matplotlib.pyplot as plt
from linalg_regressor.engine import ProjectionRegressor
from linalg_regressor.robust import RobustProjectionRegressor

# 1. Generate clean data + 1 massive outlier
np.random.seed(42)
X = np.linspace(1, 10, 20).reshape(-1, 1)
y_clean = 2.5 * X.flatten() + 1.0 + np.random.normal(0, 0.5, 20)
y_corrupted = y_clean.copy()
y_corrupted[5] = 85.0  # The outlier

# 2. Fit both models
std_model = ProjectionRegressor().fit(X, y_corrupted)
robust_model = RobustProjectionRegressor(max_trials=200, residual_threshold=2.0).fit(X, y_corrupted)

# 3. Plot the geometric breakdown
plt.figure(figsize=(10, 6))
plt.scatter(X, y_corrupted, color='red', label='Data Points (with Outlier)', zorder=5)
plt.scatter(X[5], y_corrupted[5], color='darkred', edgecolors='black', s=150, label='The Outlier (strang timestamp 00:15:28)', zorder=6)

# Plot regression lines
X_line = np.linspace(1, 10, 100).reshape(-1, 1)
plt.plot(X_line, std_model.predict(X_line), color='orange', linestyle='--', linewidth=2, label='Standard Least Squares (Skewed)')
plt.plot(X_line, robust_model.predict(X_line), color='green', linewidth=2, label='Our Robust Projection Engine (Accurate)')

plt.title("Geometric Proof: Outlier Resilience Breakdown", fontsize=14, fontweight='bold')
plt.xlabel("X (Independent Variable / Basis Space)", fontsize=12)
plt.ylabel("y (Target Vector b)", fontsize=12)
plt.ylim(0, 35) # Zoom in to see the line differences clearly
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)

# Save the plot image
plt.savefig("outlier_comparison.png", dpi=300)
print("📸 Visual geometric plot saved successfully as 'outlier_comparison.png'!")