import numpy as np
from typing import Dict

# ============================================================================
# PERFORMANCE EVALUATION FUNCTIONS
# ============================================================================


def calculate_performance_metrics(
    returns: np.ndarray, periods_per_year: int = 252
) -> Dict[str, float]:
    """
    Calculate performance metrics for portfolio returns.
    Ensures no returned metric is nan; returns 0 instead,
    and if the portfolio went bankrupt (cumulative < 1e-8), results stay at zero.

    Args:
        returns: Sequence of periodic portfolio returns.
        periods_per_year: Number of periods per year (e.g., 252 for daily, 365 for daily, 8760 for hourly).

    Returns:
        Dictionary of performance metrics.
    """
    n_periods = len(returns)
    if n_periods == 0:
        return {
            'total_return': 0,
            'annualized_return': 0,
            'volatility': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0
        }

    # Bankrupt threshold
    CUMPROD_BKR_THRESHOLD = 1e-8

    cumulative = np.cumprod(1 + returns)
    # If ever "bankrupt", all future values should stay at zero
    bankrupt_mask = cumulative < CUMPROD_BKR_THRESHOLD
    if np.any(bankrupt_mask):
        # After first true, set all to 0
        first_bkr_idx = np.argmax(bankrupt_mask)
        cumulative[first_bkr_idx:] = 0.0

    # Total return
    if np.any(cumulative == 0):
        total_return = 0
    else:
        total_return = cumulative[-1] - 1

    # Annualized return (using average periodic return)
    if n_periods == 0:
        annualized_return = 0
    else:
        avg_return = np.mean(returns)
        try:
            annualized_return = (1 + avg_return) ** periods_per_year - 1
        except Exception:
            annualized_return = 0
    if np.isnan(annualized_return):
        annualized_return = 0

    # Volatility (annualized, using custom periods_per_year)
    volatility = np.std(returns) * np.sqrt(periods_per_year)
    if np.isnan(volatility):
        volatility = 0

    # Sharpe ratio (using annualized return and volatility, risk-free rate = 0)
    sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
    if np.isnan(sharpe_ratio):
        sharpe_ratio = 0

    # Maximum drawdown
    if np.all(cumulative == 0):
        max_drawdown = 0
    else:
        running_max = np.maximum.accumulate(cumulative)
        # avoid divide by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            drawdown = np.where(running_max == 0, 0, (cumulative - running_max) / running_max)
            max_drawdown = np.min(drawdown)
        if np.isnan(max_drawdown):
            max_drawdown = 0

    return {
        'total_return': total_return if not np.isnan(total_return) else 0,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown
    }


def calculate_max_drawdown_series(returns: np.ndarray) -> np.ndarray:
    """Calculate drawdown series for plotting underwater chart."""
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    return drawdown
