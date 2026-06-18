import pytest
import numpy as np
# from linalg_regressor.engine import ProjectionRegressor
# from linalg_regressor.diagnostics import GeometricDiagnostics

def test_mathematical_orthogonality():
    """Tests if the calculated error vector is always orthogonal to column space."""
    np.random.seed(1806) # Class lecture homage seed
    X = np.random.rand(100, 3)
    y = np.random.rand(100)
    
    model = ProjectionRegressor()
    model.fit(X, y)
    
    diagnostic_results = GeometricDiagnostics.verify_orthogonality(model)
    assert diagnostic_results["is_orthogonal"] == True
    assert diagnostic_results["max_deviation"] < 1e-10

def test_idempotency():
    """Tests if P^2 = P holds true."""
    X = np.random.rand(50, 2)
    y = np.random.rand(50)
    
    model = ProjectionRegressor()
    model.fit(X, y)
    
    assert GeometricDiagnostics.verify_projection_idempotency(model) == True

def test_collinearity_handling():
    """Ensures a linear algebra failure error is safely triggered on dependent columns."""
    X = np.ones((10, 2)) # Columns 1 and 2 are perfectly identical/dependent
    y = np.random.rand(10)
    
    model = ProjectionRegressor()
    with pytest.raises(np.linalg.LinAlgError):
        model.fit(X, y)