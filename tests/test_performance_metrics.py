"""
Tests for performance_metrics module.
"""
import numpy as np
import pytest
from zetakit.performance_metrics import (
    calculate_performance_metrics,
    calculate_max_drawdown_series,
)


class TestCalculatePerformanceMetrics:
    """Test performance metrics calculation."""
    
    def test_basic_metrics(self, sample_returns):
        """Test basic metrics calculation."""
        metrics = calculate_performance_metrics(sample_returns)
        
        assert 'total_return' in metrics
        assert 'annualized_return' in metrics
        assert 'volatility' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics
        
        # All metrics should be finite (not NaN)
        for key, value in metrics.items():
            assert not np.isnan(value), f"{key} is NaN"
            assert np.isfinite(value), f"{key} is not finite"
    
    def test_empty_returns(self):
        """Test handling of empty returns array."""
        metrics = calculate_performance_metrics(np.array([]))
        
        assert metrics['total_return'] == 0
        assert metrics['annualized_return'] == 0
        assert metrics['volatility'] == 0
        assert metrics['sharpe_ratio'] == 0
        assert metrics['max_drawdown'] == 0
    
    def test_positive_returns(self):
        """Test with consistently positive returns."""
        returns = np.array([0.01, 0.02, 0.015, 0.01])
        metrics = calculate_performance_metrics(returns)
        
        assert metrics['total_return'] > 0
        assert metrics['annualized_return'] > 0
        assert metrics['max_drawdown'] <= 0  # Drawdown should be non-positive
    
    def test_negative_returns(self):
        """Test with consistently negative returns."""
        returns = np.array([-0.01, -0.02, -0.015, -0.01])
        metrics = calculate_performance_metrics(returns)
        
        assert metrics['total_return'] < 0
        assert metrics['annualized_return'] < 0
        assert metrics['max_drawdown'] < 0
    
    def test_bankrupt_portfolio(self):
        """Test handling of bankrupt portfolio."""
        # Returns that lead to near-zero cumulative
        returns = np.array([-0.5, -0.5, -0.5, -0.5])
        metrics = calculate_performance_metrics(returns)
        
        # Should handle bankruptcy gracefully
        assert not np.isnan(metrics['total_return'])
        assert not np.isnan(metrics['max_drawdown'])
    
    def test_periods_per_year(self):
        """Test different periods_per_year parameter."""
        returns = np.array([0.01] * 252)  # Daily returns
        
        metrics_daily = calculate_performance_metrics(returns, periods_per_year=252)
        metrics_monthly = calculate_performance_metrics(returns, periods_per_year=12)
        
        # Volatility should scale with sqrt(periods_per_year)
        assert metrics_daily['volatility'] <= metrics_monthly['volatility']
    
    def test_sharpe_ratio_zero_volatility(self):
        """Test Sharpe ratio with zero volatility."""
        returns = np.array([0.01, 0.01, 0.01])  # Constant returns
        metrics = calculate_performance_metrics(returns)
        
        # Sharpe should be 0 when volatility is 0
        assert metrics['sharpe_ratio'] == 0 or metrics['volatility'] == 0


class TestCalculateMaxDrawdownSeries:
    """Test drawdown series calculation."""
    
    def test_basic_drawdown(self):
        """Test basic drawdown calculation."""
        returns = np.array([0.1, -0.05, -0.1, 0.05])
        drawdown = calculate_max_drawdown_series(returns)
        
        assert len(drawdown) == len(returns)
        assert np.all(drawdown <= 0)  # Drawdowns should be non-positive
    
    def test_no_drawdown(self):
        """Test with consistently increasing portfolio."""
        returns = np.array([0.01, 0.02, 0.03, 0.04])
        drawdown = calculate_max_drawdown_series(returns)
        
        # Should have no drawdowns (all zeros or very small)
        assert np.all(drawdown >= -1e-10)
    
    def test_maximum_drawdown(self):
        """Test that maximum drawdown is captured."""
        # Create a clear drawdown pattern
        returns = np.array([0.1, 0.1, -0.2, -0.2, 0.1])
        drawdown = calculate_max_drawdown_series(returns)
        
        # Should capture the drawdown
        assert np.min(drawdown) < -0.1  # Should have significant drawdown
    
    def test_empty_returns(self):
        """Test handling of empty returns."""
        drawdown = calculate_max_drawdown_series(np.array([]))
        assert len(drawdown) == 0

