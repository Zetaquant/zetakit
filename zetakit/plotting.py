import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List
from .performance_metrics import calculate_max_drawdown_series

# ============================================================================
# PERFORMANCE PLOTTING FUNCTIONS
# ============================================================================

def plot_portfolio_composition(weights: np.ndarray, asset_names: List[str], 
                               title: str = "Portfolio Composition"):
    """Plot portfolio weights as a bar chart."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Sort by weight for better visualization
    sorted_idx = np.argsort(weights)[::-1]
    sorted_weights = weights[sorted_idx]
    sorted_names = [asset_names[i] for i in sorted_idx]
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(weights)))
    ax.bar(range(len(sorted_weights)), sorted_weights, color=colors)
    ax.set_xticks(range(len(sorted_names)))
    ax.set_xticklabels(sorted_names, rotation=45, ha='right')
    ax.set_ylabel('Weight')
    ax.set_title(title)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    return fig


def plot_cumulative_returns(returns: np.ndarray, dates: pd.DatetimeIndex, 
                title: str = "Portfolio Returns"):
    """Plot cumulative portfolio returns over time."""
    fig, ax = plt.subplots(figsize=(14, 6))
    cumulative = np.cumprod(1 + returns)
    ax.plot(dates, cumulative, linewidth=2)
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Return')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1, color='r', linestyle='--', alpha=0.5, label='Break-even')
    ax.legend()
    plt.tight_layout()
    return fig


def plot_underwater_chart(returns: np.ndarray, dates: pd.DatetimeIndex,
                         title: str = "Underwater Chart (Drawdown)"):
    """Plot underwater chart showing drawdowns."""
    fig, ax = plt.subplots(figsize=(14, 6))
    drawdown = calculate_max_drawdown_series(returns)
    ax.fill_between(dates, drawdown, 0, alpha=0.3, color='red')
    ax.plot(dates, drawdown, linewidth=1.5, color='darkred')
    ax.set_xlabel('Date')
    ax.set_ylabel('Drawdown')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([min(drawdown) * 1.1, 0.01])
    plt.tight_layout()
    return fig


def plot_return_histogram(returns: np.ndarray, title: str = "Return Distribution"):
    """Plot histogram of portfolio returns."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(returns, bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero return')
    ax.axvline(x=np.mean(returns), color='g', linestyle='--', linewidth=2, 
               label=f'Mean: {np.mean(returns):.4f}')
    ax.set_xlabel('Daily Return')
    ax.set_ylabel('Frequency')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_comprehensive_results(result_dict: Dict, asset_names: List[str], 
                              method_name: str, fold: int = None):
    """
    Create comprehensive visualization for a single result.
    
    Parameters:
    -----------
    result_dict : dict
        Single result dictionary from cross-validation
    asset_names : List[str]
        List of asset names
    method_name : str
        Name of the optimization method
    fold : int, optional
        Fold number for title
    """
    weights = result_dict['weights']
    test_returns = result_dict['test_returns']
    test_dates = result_dict['test_dates']
    metrics = result_dict['metrics']
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Title
    title_suffix = f" (Fold {fold})" if fold else ""
    fig.suptitle(f'{method_name}{title_suffix}\n'
                 f'Sharpe: {metrics["sharpe_ratio"]:.3f} | '
                 f'Return: {metrics["annualized_return"]*100:.2f}% | '
                 f'Vol: {metrics["volatility"]*100:.2f}% | '
                 f'Max DD: {metrics["max_drawdown"]*100:.2f}%', 
                 fontsize=14, fontweight='bold')
    
    # 1. Portfolio Composition
    ax1 = fig.add_subplot(gs[0, 0])
    sorted_idx = np.argsort(weights)[::-1]
    sorted_weights = weights[sorted_idx]
    sorted_names = [asset_names[i] for i in sorted_idx]
    colors = plt.cm.viridis(np.linspace(0, 1, len(weights)))
    ax1.bar(range(len(sorted_weights)), sorted_weights, color=colors)
    ax1.set_xticks(range(len(sorted_names)))
    ax1.set_xticklabels(sorted_names, rotation=45, ha='right', fontsize=8)
    ax1.set_ylabel('Weight')
    ax1.set_title('Portfolio Composition')
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Cumulative Returns
    ax2 = fig.add_subplot(gs[0, 1])
    cumulative = np.cumprod(1 + test_returns)
    ax2.plot(test_dates, cumulative, linewidth=2)
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Cumulative Return')
    ax2.set_title('Cumulative Returns')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=1, color='r', linestyle='--', alpha=0.5)
    
    # 3. Underwater Chart
    ax3 = fig.add_subplot(gs[1, :])
    drawdown = calculate_max_drawdown_series(test_returns)
    ax3.fill_between(test_dates, drawdown, 0, alpha=0.3, color='red')
    ax3.plot(test_dates, drawdown, linewidth=1.5, color='darkred')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Drawdown')
    ax3.set_title('Underwater Chart (Drawdown)')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([min(drawdown) * 1.1, 0.01])
    
    # 4. Return Histogram
    ax4 = fig.add_subplot(gs[2, :])
    ax4.hist(test_returns, bins=50, alpha=0.7, edgecolor='black')
    ax4.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero return')
    ax4.axvline(x=np.mean(test_returns), color='g', linestyle='--', linewidth=2, 
               label=f'Mean: {np.mean(test_returns):.4f}')
    ax4.set_xlabel('Daily Return')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Return Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig
