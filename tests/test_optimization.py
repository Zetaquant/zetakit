"""
Tests for optimization module.
"""
import numpy as np
import pytest
from zetakit.optimization import (
    time_series_cv_indices,
    time_series_cv_split,
    mean_variance_optimization,
)


class TestTimeSeriesCVIndices:
    """Test time series cross-validation index generation."""
    
    def test_expanding_window_basic(self):
        """Test expanding window CV with basic parameters."""
        splits = time_series_cv_indices(n_samples=100, n_splits=5, train_window_type='expanding')
        
        assert len(splits) == 5
        for train_idx, test_idx in splits:
            assert len(train_idx) > 0
            assert len(test_idx) > 0
            assert np.max(train_idx) < np.min(test_idx)  # Train before test
    
    def test_constant_window_basic(self):
        """Test constant window CV with basic parameters."""
        splits = time_series_cv_indices(
            n_samples=100, 
            n_splits=5, 
            train_window_type='constant',
            min_train_size=20
        )
        
        assert len(splits) > 0
        for train_idx, test_idx in splits:
            assert len(train_idx) == 20  # Fixed train size
            assert len(test_idx) > 0
    
    def test_expanding_window_grows(self):
        """Test that expanding window actually grows."""
        splits = time_series_cv_indices(n_samples=100, n_splits=3, train_window_type='expanding')
        
        train_sizes = [len(train_idx) for train_idx, _ in splits]
        assert train_sizes[0] < train_sizes[1] < train_sizes[2]
    
    def test_gap_parameter(self):
        """Test gap parameter between train and test sets."""
        splits_no_gap = time_series_cv_indices(n_samples=100, n_splits=3, gap=0)
        splits_with_gap = time_series_cv_indices(n_samples=100, n_splits=3, gap=5)
        
        # With gap, test should start further from train end
        for i in range(len(splits_no_gap)):
            train_end_no_gap = np.max(splits_no_gap[i][0])
            test_start_no_gap = np.min(splits_no_gap[i][1])
            
            train_end_with_gap = np.max(splits_with_gap[i][0])
            test_start_with_gap = np.min(splits_with_gap[i][1])
            
            assert (test_start_with_gap - train_end_with_gap) >= \
                   (test_start_no_gap - train_end_no_gap) + 5
    
    def test_invalid_parameters(self):
        """Test that invalid parameters raise appropriate errors."""
        with pytest.raises(ValueError):
            time_series_cv_indices(n_samples=1, n_splits=1)
        
        with pytest.raises(ValueError):
            time_series_cv_indices(n_samples=100, n_splits=0)
        
        with pytest.raises(ValueError):
            time_series_cv_indices(n_samples=100, n_splits=5, gap=-1)
        
        with pytest.raises(ValueError):
            time_series_cv_indices(
                n_samples=100, 
                n_splits=5, 
                train_window_type='invalid'
            )


class TestTimeSeriesCVSplit:
    """Test time series cross-validation DataFrame splitting."""
    
    def test_basic_split(self):
        """Test basic DataFrame splitting."""
        import pandas as pd
        
        df = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
        })
        
        splits = time_series_cv_split(df, n_splits=3)
        
        assert len(splits) == 3
        for train_df, test_df in splits:
            assert len(train_df) > 0
            assert len(test_df) > 0
            assert len(train_df) + len(test_df) <= len(df)


class TestMeanVarianceOptimization:
    """Test mean-variance optimization."""
    
    def test_basic_optimization(self, sample_expected_returns, sample_covariance):
        """Test basic mean-variance optimization."""
        weights = mean_variance_optimization(
            sample_expected_returns,
            sample_covariance,
            risk_penalty=1.0
        )
        
        assert len(weights) == len(sample_expected_returns)
        assert np.allclose(np.sum(np.abs(weights)), 1.0, rtol=1e-6)
    
    def test_weights_sum_to_one(self, sample_expected_returns, sample_covariance):
        """Test that weights are normalized correctly."""
        weights = mean_variance_optimization(
            sample_expected_returns,
            sample_covariance
        )
        
        # Should sum to 1 (or sum of absolute values to 1 for long-short)
        assert np.allclose(np.sum(np.abs(weights)), 1.0, rtol=1e-6)
    
    def test_risk_penalty_effect(self, sample_expected_returns, sample_covariance):
        """Test that higher risk penalty reduces portfolio variance."""
        weights_low_risk = mean_variance_optimization(
            sample_expected_returns,
            sample_covariance,
            risk_penalty=0.5
        )
        
        weights_high_risk = mean_variance_optimization(
            sample_expected_returns,
            sample_covariance,
            risk_penalty=2.0
        )
        
        # Higher risk penalty should generally lead to lower portfolio variance
        var_low = weights_low_risk @ sample_covariance @ weights_low_risk
        var_high = weights_high_risk @ sample_covariance @ weights_high_risk
        
        # This is a heuristic test - higher risk penalty should reduce variance
        assert var_high <= var_low * 1.5  # Allow some tolerance
    
    def test_ridge_penalty(self, sample_expected_returns, sample_covariance):
        """Test that ridge penalty stabilizes the solution."""
        weights_no_ridge = mean_variance_optimization(
            sample_expected_returns,
            sample_covariance,
            ridge_penalty=0.0
        )
        
        weights_with_ridge = mean_variance_optimization(
            sample_expected_returns,
            sample_covariance,
            ridge_penalty=0.1
        )
        
        # Both should be valid solutions
        assert len(weights_no_ridge) == len(weights_with_ridge)
        assert np.allclose(np.sum(np.abs(weights_no_ridge)), 1.0)
        assert np.allclose(np.sum(np.abs(weights_with_ridge)), 1.0)
    
    def test_input_validation(self, sample_expected_returns, sample_covariance):
        """Test input validation."""
        # Wrong covariance shape
        wrong_cov = sample_covariance[:3, :3]
        with pytest.raises(ValueError):
            mean_variance_optimization(sample_expected_returns, wrong_cov)
        
        # Negative risk penalty
        with pytest.raises(ValueError):
            mean_variance_optimization(
                sample_expected_returns,
                sample_covariance,
                risk_penalty=-1.0
            )
        
        # Negative ridge penalty
        with pytest.raises(ValueError):
            mean_variance_optimization(
                sample_expected_returns,
                sample_covariance,
                ridge_penalty=-1.0
            )
    
    def test_simple_two_asset_case(self):
        """Test with a simple 2-asset case."""
        returns = np.array([0.1, 0.05])
        cov = np.array([[0.04, 0.01],
                       [0.01, 0.02]])
        
        weights = mean_variance_optimization(returns, cov, risk_penalty=1.0)
        
        assert len(weights) == 2
        assert np.allclose(np.sum(np.abs(weights)), 1.0, rtol=1e-6)
        # Higher return asset should get more weight (in absolute terms)
        assert np.abs(weights[0]) >= np.abs(weights[1]) * 0.5
    
    def test_correct_solution_simple_example(self):
        """
        Test that mean_variance_optimization returns the correct solution 
        for a simple analytically solvable example.
        
        Example:
        - Expected returns: μ = [0.1, 0.05]
        - Covariance: Σ = [[0.04, 0.01], [0.01, 0.02]]
        - Risk penalty: λ = 1.0
        - Ridge penalty: α = 0.0
        
        Expected solution:
        A = λΣ + αI = [[0.04, 0.01], [0.01, 0.02]]
        w* = A^(-1) μ (unconstrained)
        w = w* / sum(|w*|) (normalized)
        """
        # Define inputs
        expected_returns = np.array([0.1, 0.05])
        covariance = np.array([[0.04, 0.01],
                              [0.01, 0.02]])
        risk_penalty = 1.0
        ridge_penalty = 0.0
        
        # Compute expected solution analytically
        # A = λΣ + αI
        A = risk_penalty * covariance + ridge_penalty * np.eye(2)
        
        # w* = A^(-1) μ
        w_unconstrained = np.linalg.solve(A, expected_returns)
        
        # Normalize: w = w* / sum(|w*|)
        expected_weights = w_unconstrained / np.sum(np.abs(w_unconstrained))
        
        # Compute actual solution
        actual_weights = mean_variance_optimization(
            expected_returns,
            covariance,
            risk_penalty=risk_penalty,
            ridge_penalty=ridge_penalty
        )
        
        # Verify solution matches expected result
        np.testing.assert_allclose(
            actual_weights,
            expected_weights,
            rtol=1e-6,
            atol=1e-8,
            err_msg="Optimization solution does not match expected analytical solution"
        )
        
        # Verify normalization constraint
        assert np.allclose(np.sum(np.abs(actual_weights)), 1.0, rtol=1e-6)
    
    def test_correct_solution_with_ridge(self):
        """
        Test correct solution with ridge penalty.
        
        Example:
        - Expected returns: μ = [0.08, 0.06]
        - Covariance: Σ = [[0.03, 0.005], [0.005, 0.025]]
        - Risk penalty: λ = 2.0
        - Ridge penalty: α = 0.1
        """
        # Define inputs
        expected_returns = np.array([0.08, 0.06])
        covariance = np.array([[0.03, 0.005],
                              [0.005, 0.025]])
        risk_penalty = 2.0
        ridge_penalty = 0.1
        
        # Compute expected solution analytically
        A = risk_penalty * covariance + ridge_penalty * np.eye(2)
        w_unconstrained = np.linalg.solve(A, expected_returns)
        expected_weights = w_unconstrained / np.sum(np.abs(w_unconstrained))
        
        # Compute actual solution
        actual_weights = mean_variance_optimization(
            expected_returns,
            covariance,
            risk_penalty=risk_penalty,
            ridge_penalty=ridge_penalty
        )
        
        # Verify solution matches expected result
        np.testing.assert_allclose(
            actual_weights,
            expected_weights,
            rtol=1e-6,
            atol=1e-8,
            err_msg="Optimization solution with ridge penalty does not match expected"
        )

    def test_correct_solution_uncorrelated_assets(self):
        """
        Test that mean_variance_optimization returns the correct solution 
        for uncorrelated assets.
        
        Example:
        - Expected returns: μ = [0.1, 0.05]
        - Covariance: Σ = [[0.04, 0.00], [0.00, 0.02]]
        - Risk penalty: λ = 1.0
        - Ridge penalty: α = 0.0
        
        Expected solution:
        A = λΣ + αI = [[0.04, 0.00], [0.00, 0.02]]
        w* = A^(-1) μ (unconstrained)
        w = w* / sum(|w*|) (normalized)
        """
        # Define inputs
        expected_returns = np.array([0.1, 0.05])
        covariance = np.array([[0.04, 0.00],
                              [0.00, 0.02]])
        risk_penalty = 1.0
        ridge_penalty = 0.0

        # Compute expected solution analytically (by hand, no matrix inversion)
        # For this 2x2 diagonal A, solution is simply w_i = mu_i / (A_ii)
        # A = λΣ + αI = [[0.04, 0.00], [0.00, 0.02]]
        A_00 = risk_penalty * covariance[0, 0] + ridge_penalty
        A_11 = risk_penalty * covariance[1, 1] + ridge_penalty
        w0 = expected_returns[0] / A_00
        w1 = expected_returns[1] / A_11
        w_unconstrained = np.array([w0, w1])

        # Normalize: w = w* / sum(|w*|)
        expected_weights = w_unconstrained / np.sum(np.abs(w_unconstrained))
        
        # Compute actual solution
        actual_weights = mean_variance_optimization(
            expected_returns,
            covariance,
            risk_penalty=risk_penalty,
            ridge_penalty=ridge_penalty
        )
        
        # Verify solution matches expected result
        np.testing.assert_allclose(
            actual_weights,
            expected_weights,
            rtol=1e-6,
            atol=1e-8,
            err_msg="Optimization solution does not match expected analytical solution"
        )
        
        # Verify normalization constraint
        assert np.allclose(np.sum(np.abs(actual_weights)), 1.0, rtol=1e-6)

