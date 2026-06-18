import numpy as np
from .engine import ProjectionRegressor

class RobustProjectionRegressor:
    """
    An outlier-resilient regression engine that utilizes RANSAC random 
    sampling combined with our custom ProjectionRegressor engine.
    """
    def __init__(self, max_trials=100, residual_threshold=1.5):
        self.max_trials = max_trials
        self.residual_threshold = residual_threshold
        self.best_model = None
        self.inlier_mask = None

    def fit(self, X, y):
        n_samples = X.shape[0]
        # Minimum samples required to define a line uniquely is 2
        min_samples = X.shape[1] + 1 
        
        best_inlier_count = 0
        best_inlier_mask = None
        
        for _ in range(self.max_trials):
            # 1. Randomly sample a subset of data points
            sub_indices = np.random.choice(n_samples, min_samples, replace=False)
            X_sub, y_sub = X[sub_indices], y[sub_indices]
            
            # 2. Fit standard projection model on this subset
            try:
                trial_model = ProjectionRegressor()
                trial_model.fit(X_sub, y_sub)
            except np.linalg.LinAlgError:
                continue # Skip if sample yields a singular matrix
                
            # 3. Evaluate errors across the entire dataset
            predictions = trial_model.predict(X)
            residuals = np.abs(y.reshape(-1, 1) - predictions.reshape(-1, 1)).flatten()
            
            # 4. Count how many points fall within the allowed error margin
            current_inlier_mask = residuals < self.residual_threshold
            inlier_count = np.sum(current_inlier_mask)
            
            # 5. Keep track of the model with the most consensus (inliers)
            if inlier_count > best_inlier_count:
                best_inlier_count = inlier_count
                best_inlier_mask = current_inlier_mask
                
        if best_inlier_mask is None:
            raise RuntimeError("Robust estimation failed to find a valid consensus set.")
            
        # Refit final model on the entire consensus set of inliers
        self.inlier_mask = best_inlier_mask
        self.best_model = ProjectionRegressor()
        self.best_model.fit(X[self.inlier_mask], y[self.inlier_mask])
        return self

    def predict(self, X):
        return self.best_model.predict(X)