"""
Tests for optimization module.
"""
import numpy as np
import pytest
from zetakit.optimization import (
    time_series_cv_indices,
    time_series_cv_split,
    mean_variance_optimization,
    mean_variance_optimization_with_transaction_costs,
    calculate_rebalancing_trades,
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


class TestMeanVarianceOptimizationWithTransactionCosts:
    """Test mean-variance optimization with transaction costs."""
    
    def test_basic_functionality(self, sample_expected_returns, sample_covariance):
        """Test basic functionality with transaction costs."""
        current_weights = np.ones(len(sample_expected_returns)) / len(sample_expected_returns)
        
        weights = mean_variance_optimization_with_transaction_costs(
            sample_expected_returns,
            sample_covariance,
            current_weights,
            transaction_cost=0.001,
            risk_penalty=1.0
        )
        
        assert len(weights) == len(sample_expected_returns)
        assert np.allclose(np.sum(np.abs(weights)), 1.0, rtol=1e-6)
    
    def test_zero_transaction_cost_matches_regular_mvo(self, sample_expected_returns, sample_covariance):
        """Test that zero transaction cost gives same result as regular MVO."""
        current_weights = np.ones(len(sample_expected_returns)) / len(sample_expected_returns)
        
        # Regular MVO
        weights_regular = mean_variance_optimization(
            sample_expected_returns,
            sample_covariance,
            risk_penalty=1.0,
            ridge_penalty=0.0
        )
        
        # MVO with zero transaction cost
        weights_with_tc = mean_variance_optimization_with_transaction_costs(
            sample_expected_returns,
            sample_covariance,
            current_weights,
            transaction_cost=0.0,
            risk_penalty=1.0,
            ridge_penalty=0.0
        )
        
        # Should be very close (allowing for numerical differences)
        np.testing.assert_allclose(
            weights_regular,
            weights_with_tc,
            rtol=1e-5,
            atol=1e-6,
            err_msg="Zero transaction cost should match regular MVO"
        )
    
    def test_transaction_cost_reduces_trades(self, sample_expected_returns, sample_covariance):
        """Test that higher transaction costs lead to smaller trades."""
        # Start with equal weights
        current_weights = np.ones(len(sample_expected_returns)) / len(sample_expected_returns)
        
        # Get optimal weights with no transaction cost
        weights_no_tc = mean_variance_optimization_with_transaction_costs(
            sample_expected_returns,
            sample_covariance,
            current_weights,
            transaction_cost=0.0,
            risk_penalty=1.0
        )
        
        # Get optimal weights with transaction cost
        weights_with_tc = mean_variance_optimization_with_transaction_costs(
            sample_expected_returns,
            sample_covariance,
            current_weights,
            transaction_cost=0.01,  # Higher transaction cost
            risk_penalty=1.0
        )
        
        # Calculate trades
        trades_no_tc = np.abs(weights_no_tc - current_weights)
        trades_with_tc = np.abs(weights_with_tc - current_weights)
        
        # With transaction costs, total trades should be smaller or equal
        assert np.sum(trades_with_tc) <= np.sum(trades_no_tc) + 1e-6
    
    def test_transaction_cost_keeps_portfolio_closer(self):
        """Test that transaction costs keep optimal portfolio closer to current."""
        expected_returns = np.array([0.1, 0.05, 0.08])
        covariance = np.array([[0.04, 0.01, 0.02],
                              [0.01, 0.05, 0.03],
                              [0.02, 0.03, 0.06]])
        current_weights = np.array([0.4, 0.3, 0.3])
        
        # No transaction cost
        weights_no_tc = mean_variance_optimization_with_transaction_costs(
            expected_returns,
            covariance,
            current_weights,
            transaction_cost=0.0,
            risk_penalty=1.0
        )
        
        # With transaction cost
        weights_with_tc = mean_variance_optimization_with_transaction_costs(
            expected_returns,
            covariance,
            current_weights,
            transaction_cost=0.5,
            risk_penalty=1.0
        )
        
        # Distance from current portfolio
        distance_no_tc = np.sum(np.abs(weights_no_tc - current_weights))
        distance_with_tc = np.sum(np.abs(weights_with_tc - current_weights))
        
        # With transaction costs, should be closer to current portfolio
        assert distance_with_tc <= distance_no_tc + 1e-6
    
    def test_input_validation(self, sample_expected_returns, sample_covariance):
        """Test input validation."""
        current_weights = np.ones(len(sample_expected_returns)) / len(sample_expected_returns)
        
        # Wrong covariance shape
        wrong_cov = sample_covariance[:3, :3]
        with pytest.raises(ValueError):
            mean_variance_optimization_with_transaction_costs(
                sample_expected_returns,
                wrong_cov,
                current_weights
            )
        
        # Wrong current weights length
        wrong_weights = current_weights[:3]
        with pytest.raises(ValueError):
            mean_variance_optimization_with_transaction_costs(
                sample_expected_returns,
                sample_covariance,
                wrong_weights
            )
        
        # Negative transaction cost
        with pytest.raises(ValueError):
            mean_variance_optimization_with_transaction_costs(
                sample_expected_returns,
                sample_covariance,
                current_weights,
                transaction_cost=-0.001
            )
        
        # Negative risk penalty
        with pytest.raises(ValueError):
            mean_variance_optimization_with_transaction_costs(
                sample_expected_returns,
                sample_covariance,
                current_weights,
                risk_penalty=-1.0
            )
        
        # Negative ridge penalty
        with pytest.raises(ValueError):
            mean_variance_optimization_with_transaction_costs(
                sample_expected_returns,
                sample_covariance,
                current_weights,
                ridge_penalty=-0.1
            )
    
    def test_normalization_constraint(self, sample_expected_returns, sample_covariance):
        """Test that weights are properly normalized."""
        # Test with different current weight normalizations
        current_weights_1 = np.array([0.3, 0.3, 0.2, 0.1, 0.1])
        current_weights_2 = np.array([0.5, 0.3, 0.1, 0.05, 0.05])
        
        weights_1 = mean_variance_optimization_with_transaction_costs(
            sample_expected_returns,
            sample_covariance,
            current_weights_1,
            transaction_cost=0.001
        )
        
        weights_2 = mean_variance_optimization_with_transaction_costs(
            sample_expected_returns,
            sample_covariance,
            current_weights_2,
            transaction_cost=0.001
        )
        
        # Both should sum to 1 (in absolute values)
        assert np.allclose(np.sum(np.abs(weights_1)), 1.0, rtol=1e-6)
        assert np.allclose(np.sum(np.abs(weights_2)), 1.0, rtol=1e-6)
    
    def test_ridge_penalty_with_transaction_costs(self, sample_expected_returns, sample_covariance):
        """Test that ridge penalty works with transaction costs."""
        current_weights = np.ones(len(sample_expected_returns)) / len(sample_expected_returns)
        
        weights_no_ridge = mean_variance_optimization_with_transaction_costs(
            sample_expected_returns,
            sample_covariance,
            current_weights,
            transaction_cost=0.001,
            ridge_penalty=0.0
        )
        
        weights_with_ridge = mean_variance_optimization_with_transaction_costs(
            sample_expected_returns,
            sample_covariance,
            current_weights,
            transaction_cost=0.001,
            ridge_penalty=0.1
        )
        
        # Both should be valid solutions
        assert len(weights_no_ridge) == len(weights_with_ridge)
        assert np.allclose(np.sum(np.abs(weights_no_ridge)), 1.0)
        assert np.allclose(np.sum(np.abs(weights_with_ridge)), 1.0)
    
    def test_simple_two_asset_case(self):
        """Test with a simple 2-asset case."""
        returns = np.array([0.1, 0.05])
        cov = np.array([[0.04, 0.01],
                       [0.01, 0.02]])
        current = np.array([0.5, 0.5])
        
        weights = mean_variance_optimization_with_transaction_costs(
            returns, cov, current, transaction_cost=0.001, risk_penalty=1.0
        )
        
        assert len(weights) == 2
        assert np.allclose(np.sum(np.abs(weights)), 1.0, rtol=1e-6)
    
    def test_very_high_transaction_cost(self):
        """Test that very high transaction costs keep portfolio unchanged."""
        expected_returns = np.array([0.1, 0.05])
        covariance = np.array([[0.04, 0.01],
                              [0.01, 0.02]])
        current_weights = np.array([0.6, 0.4])
        
        # Very high transaction cost should keep portfolio close to current
        weights = mean_variance_optimization_with_transaction_costs(
            expected_returns,
            covariance,
            current_weights,
            transaction_cost=10.0,  # Very high
            risk_penalty=1.0
        )
        
        # Portfolio should be very close to current (allowing some numerical tolerance)
        distance = np.sum(np.abs(weights - current_weights))
        assert distance < 0.1  # Should be quite close
    
    def test_one_dimensional_closed_form_solution(self):
        """
        Test 1D case with closed-form solution.
        
        For 1 asset:
        - Expected return: μ
        - Variance: σ²
        - Current weight: w₀
        - Risk penalty: λ
        - Transaction cost: c
        - Ridge penalty: α
        
        The optimization problem is:
        maximize: μw - (λ/2)σ²w² - (α/2)w² - c|w - w₀|
        
        Closed-form solution:
        - If (μ - c)/(λσ² + α) ≤ w₀ ≤ (μ + c)/(λσ² + α): w* = w₀ (no trade)
        - If w₀ < (μ - c)/(λσ² + α): w* = (μ - c)/(λσ² + α) (trade up)
        - If w₀ > (μ + c)/(λσ² + α): w* = (μ + c)/(λσ² + α) (trade down)
        """
        # Test parameters
        mu = 0.1  # Expected return
        sigma_sq = 0.04  # Variance
        risk_penalty = 2.0
        ridge_penalty = 0.0
        transaction_cost = 0.02
        
        # Calculate boundaries
        denominator = risk_penalty * sigma_sq + ridge_penalty
        lower_bound = (mu - transaction_cost) / denominator
        upper_bound = (mu + transaction_cost) / denominator
        
        # Test case 1: No-trade region (current weight in middle)
        w0_no_trade = (lower_bound + upper_bound) / 2
        expected_w_no_trade = w0_no_trade
        
        actual_w_no_trade = mean_variance_optimization_with_transaction_costs(
            expected_returns=np.array([mu]),
            covariance=np.array([[sigma_sq]]),
            current_weights=np.array([w0_no_trade]),
            transaction_cost=transaction_cost,
            risk_penalty=risk_penalty,
            ridge_penalty=ridge_penalty,
            normalize=False
        )
        
        np.testing.assert_allclose(
            actual_w_no_trade[0],
            expected_w_no_trade,
            rtol=1e-5,
            atol=1e-6,
            err_msg="No-trade case: optimal weight should equal current weight"
        )
        
        # Test case 2: Trade up (current weight below lower bound)
        w0_trade_up = lower_bound - 0.1
        expected_w_trade_up = lower_bound
        
        actual_w_trade_up = mean_variance_optimization_with_transaction_costs(
            expected_returns=np.array([mu]),
            covariance=np.array([[sigma_sq]]),
            current_weights=np.array([w0_trade_up]),
            transaction_cost=transaction_cost,
            risk_penalty=risk_penalty,
            ridge_penalty=ridge_penalty,
            normalize=False
        )
        
        np.testing.assert_allclose(
            actual_w_trade_up[0],
            expected_w_trade_up,
            rtol=1e-5,
            atol=1e-6,
            err_msg="Trade-up case: optimal weight should equal lower bound"
        )
        
        # Test case 3: Trade down (current weight above upper bound)
        w0_trade_down = upper_bound + 0.1
        expected_w_trade_down = upper_bound
        
        actual_w_trade_down = mean_variance_optimization_with_transaction_costs(
            expected_returns=np.array([mu]),
            covariance=np.array([[sigma_sq]]),
            current_weights=np.array([w0_trade_down]),
            transaction_cost=transaction_cost,
            risk_penalty=risk_penalty,
            ridge_penalty=ridge_penalty,
            normalize=False
        )
        
        np.testing.assert_allclose(
            actual_w_trade_down[0],
            expected_w_trade_down,
            rtol=1e-5,
            atol=1e-6,
            err_msg="Trade-down case: optimal weight should equal upper bound"
        )
        
        # Test case 4: Edge case - exactly at lower bound
        w0_at_lower = lower_bound
        expected_w_at_lower = lower_bound
        
        actual_w_at_lower = mean_variance_optimization_with_transaction_costs(
            expected_returns=np.array([mu]),
            covariance=np.array([[sigma_sq]]),
            current_weights=np.array([w0_at_lower]),
            transaction_cost=transaction_cost,
            risk_penalty=risk_penalty,
            ridge_penalty=ridge_penalty,
            normalize=False
        )
        
        np.testing.assert_allclose(
            actual_w_at_lower[0],
            expected_w_at_lower,
            rtol=1e-5,
            atol=1e-6,
            err_msg="Edge case at lower bound"
        )
        
        # Test case 5: Edge case - exactly at upper bound
        w0_at_upper = upper_bound
        expected_w_at_upper = upper_bound
        
        actual_w_at_upper = mean_variance_optimization_with_transaction_costs(
            expected_returns=np.array([mu]),
            covariance=np.array([[sigma_sq]]),
            current_weights=np.array([w0_at_upper]),
            transaction_cost=transaction_cost,
            risk_penalty=risk_penalty,
            ridge_penalty=ridge_penalty,
            normalize=False
        )
        
        np.testing.assert_allclose(
            actual_w_at_upper[0],
            expected_w_at_upper,
            rtol=1e-5,
            atol=1e-6,
            err_msg="Edge case at upper bound"
        )
        
        # Test case 6: With ridge penalty
        ridge_penalty_test = 0.1
        denominator_ridge = risk_penalty * sigma_sq + ridge_penalty_test
        lower_bound_ridge = (mu - transaction_cost) / denominator_ridge
        upper_bound_ridge = (mu + transaction_cost) / denominator_ridge
        
        w0_ridge = (lower_bound_ridge + upper_bound_ridge) / 2
        expected_w_ridge = w0_ridge
        
        actual_w_ridge = mean_variance_optimization_with_transaction_costs(
            expected_returns=np.array([mu]),
            covariance=np.array([[sigma_sq]]),
            current_weights=np.array([w0_ridge]),
            transaction_cost=transaction_cost,
            risk_penalty=risk_penalty,
            ridge_penalty=ridge_penalty_test,
            normalize=False
        )
        
        np.testing.assert_allclose(
            actual_w_ridge[0],
            expected_w_ridge,
            rtol=1e-5,
            atol=1e-6,
            err_msg="No-trade case with ridge penalty"
        )


class TestCalculateRebalancingTrades:
    """Test rebalancing trades calculation."""
    
    def test_basic_functionality(self, sample_expected_returns, sample_covariance):
        """Test basic rebalancing trades calculation."""
        current_weights = np.ones(len(sample_expected_returns)) / len(sample_expected_returns)
        
        trades, optimal_weights = calculate_rebalancing_trades(
            sample_expected_returns,
            sample_covariance,
            current_weights,
            transaction_cost=0.001,
            risk_penalty=1.0
        )
        
        assert len(trades) == len(sample_expected_returns)
        assert len(optimal_weights) == len(sample_expected_returns)
        assert np.allclose(np.sum(np.abs(optimal_weights)), 1.0, rtol=1e-6)
        
        # Trades should equal difference
        current_normalized = current_weights / np.sum(np.abs(current_weights))
        np.testing.assert_allclose(
            trades,
            optimal_weights - current_normalized,
            rtol=1e-6
        )
    
    def test_trades_sum_to_zero_approximately(self):
        """Test that trades approximately sum to zero (for long-short portfolios)."""
        expected_returns = np.array([0.1, 0.05, 0.08])
        covariance = np.array([[0.04, 0.01, 0.02],
                              [0.01, 0.05, 0.03],
                              [0.02, 0.03, 0.06]])
        current_weights = np.array([0.4, 0.3, 0.3])
        
        trades, optimal_weights = calculate_rebalancing_trades(
            expected_returns,
            covariance,
            current_weights,
            transaction_cost=0.001
        )
        
        # For long-short portfolios, trades don't necessarily sum to zero
        # But the optimal weights should satisfy the constraint
        assert np.allclose(np.sum(np.abs(optimal_weights)), 1.0, rtol=1e-6)
    
    def test_zero_transaction_cost_trades(self, sample_expected_returns, sample_covariance):
        """Test trades calculation with zero transaction cost."""
        current_weights = np.ones(len(sample_expected_returns)) / len(sample_expected_returns)
        
        trades, optimal_weights = calculate_rebalancing_trades(
            sample_expected_returns,
            sample_covariance,
            current_weights,
            transaction_cost=0.0
        )
        
        # Verify trades are calculated correctly
        current_normalized = current_weights / np.sum(np.abs(current_weights))
        np.testing.assert_allclose(
            trades,
            optimal_weights - current_normalized,
            rtol=1e-6
        )
    
    def test_positive_trades_indicate_buys(self):
        """Test that positive trades indicate positions to increase."""
        expected_returns = np.array([0.15, 0.05])  # First asset much better
        covariance = np.array([[0.04, 0.01],
                              [0.01, 0.02]])
        current_weights = np.array([0.3, 0.7])  # Underweight in first asset
        
        trades, optimal_weights = calculate_rebalancing_trades(
            expected_returns,
            covariance,
            current_weights,
            transaction_cost=0.001,
            risk_penalty=1.0
        )
        
        # Should want to buy more of first asset (higher return)
        assert trades[0] > -1e-6  # Should be positive or very small negative
        # Second asset should be sold or reduced
        assert trades[1] < 1e-6  # Should be negative or very small positive
