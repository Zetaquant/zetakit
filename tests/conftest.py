"""
Pytest configuration and shared fixtures.
"""
import numpy as np
import pytest


@pytest.fixture
def sample_returns():
    """Generate sample returns for testing."""
    np.random.seed(42)
    return np.random.randn(100) * 0.02  # 2% daily volatility


@pytest.fixture
def sample_covariance():
    """Generate a sample covariance matrix."""
    np.random.seed(42)
    n = 5
    A = np.random.randn(n, n)
    return A @ A.T + 0.01 * np.eye(n)  # Ensure positive definite


@pytest.fixture
def sample_expected_returns():
    """Generate sample expected returns."""
    np.random.seed(42)
    return np.random.randn(5) * 0.1 + 0.05  # Mean 5%, std 10%

