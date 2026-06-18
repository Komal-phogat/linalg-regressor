import numpy as np

class GeometricDiagnostics:
    """
    Mathematical verification tool to validate the geometric properties 
    of the underlying projection matrices and vector spaces.
    """
    @staticmethod
    def verify_orthogonality(model, tolerance=1e-10):
        """
        Validates Strang's core proof: The error vector 'e' must be completely 
        orthogonal to the column space of A. Meaning: A^T * e = 0.
        """
        if model.e is None or model.A is None:
            raise ValueError("Model must be fitted before running diagnostics.")
            
        # Calculate A^T * e
        orthogonality_check = model.A.T @ model.e
        
        # Check if all elements are close to absolute zero within tolerance
        is_orthogonal = np.allclose(orthogonality_check, 0, atol=tolerance)
        
        return {
            "is_orthogonal": is_orthogonal,
            "raw_dot_products": orthogonality_check.flatten(),
            "max_deviation": np.max(np.abs(orthogonality_check))
        }

    @staticmethod
    def verify_projection_idempotency(model, tolerance=1e-10):
        """
        Validates that the projection matrix is idempotent (P^2 = P).
        Projecting an already projected vector shouldn't change it.
        """
        if model.hat_matrix is None:
            raise ValueError("Model must be fitted before running diagnostics.")
            
        P = model.hat_matrix
        P_squared = P @ P
        
        is_idempotent = np.allclose(P_squared, P, atol=tolerance)
        return is_idempotent
        