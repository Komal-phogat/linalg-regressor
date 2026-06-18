import numpy as np

class ProjectionRegressor:
    """
    Linear regression engine implemented via vector space projections 
    and the Normal Equations, as taught in MIT 18.06 Lecture 16.
    """
    def __init__(self):
        self.beta = None          # Coefficients/Weights (C and D in the lecture)
        self.hat_matrix = None     # The explicit Projection Matrix P
        self.A = None             # Design matrix (basis for column space)
        self.b = None             # Target vector
        self.p = None             # Projection vector (predicted values in column space)
        self.e = None             # Error vector (residuals perpendicular to column space)

    def _build_design_matrix(self, X):
        """Prepends a column of ones to handle the intercept."""
        ones = np.ones((X.shape[0], 1))
        return np.hstack([ones, X])

    def fit(self, X, y):
        """
        Fits the model by projecting y onto the column space of X.
        """
        # Ensure inputs are correct numpy shapes
        self.b = y.reshape(-1, 1)
        self.A = self._build_design_matrix(X)
        
        # Academic Guardrail: Ensure columns are independent so A^T * A is invertible
        rank = np.linalg.matrix_rank(self.A)
        num_features = self.A.shape[1]
        if rank < num_features:
            raise np.linalg.LinAlgError(
                f"Multi-collinearity detected! Matrix rank ({rank}) is less than "
                f"the number of features ({num_features}). A^T * A is not invertible."
            )
            
        # Compute the Normal Equations: beta = (A^T * A)^-1 * A^T * b
        ATA = self.A.T @ self.A
        ATb = self.A.T @ self.b
        self.beta = np.linalg.inv(ATA) @ ATb
        
        # Calculate the fundamental components from the lecture
        # P = A * (A^T * A)^-1 * A^T
        self.hat_matrix = self.A @ np.linalg.inv(ATA) @ self.A.T
        
        # p = P * b (The projection vector)
        self.p = self.hat_matrix @ self.b
        
        # e = b - p = (I - P) * b (The error vector)
        self.e = self.b - self.p
        
        return self

    def predict(self, X):
        """Predicts targets for new data points using computed betas."""
        if self.beta is None:
            raise ValueError("Model must be fitted before making predictions.")
        A_new = self._build_design_matrix(X)
        return A_new @ self.beta