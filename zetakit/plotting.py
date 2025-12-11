import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union, Literal
from scipy import stats
try:
    from scipy.interpolate import UnivariateSpline
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    UnivariateSpline = None

from .performance_metrics import calculate_max_drawdown_series

# ============================================================================
# GENERAL PLOTTING FUNCTIONS
# ============================================================================

def plot_histogram(data: Union[np.ndarray, pd.Series], 
                   bins: Union[int, str] = 'auto',
                   xlabel: str = 'Value',
                   ylabel: str = 'Frequency',
                   title: str = 'Histogram',
                   figsize: tuple = (10, 6),
                   color: str = 'steelblue',
                   alpha: float = 0.7,
                   edgecolor: str = 'black',
                   show_mean: bool = True,
                   show_median: bool = True,
                   density: bool = False):
    """
    Plot a histogram with optional statistical markers.
    
    Parameters:
    -----------
    data : np.ndarray or pd.Series
        Data to plot
    bins : int or str, default 'auto'
        Number of bins or binning strategy (passed to np.histogram)
    xlabel : str, default 'Value'
        Label for x-axis
    ylabel : str, default 'Frequency'
        Label for y-axis
    title : str, default 'Histogram'
        Plot title
    figsize : tuple, default (10, 6)
        Figure size (width, height)
    color : str, default 'steelblue'
        Histogram color
    alpha : float, default 0.7
        Transparency level
    edgecolor : str, default 'black'
        Edge color for bars
    show_mean : bool, default True
        Whether to show mean line
    show_median : bool, default True
        Whether to show median line
    density : bool, default False
        If True, normalize to form a probability density
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Convert to numpy array if needed
    if isinstance(data, pd.Series):
        data = data.values
    
    # Remove NaN values
    data = data[~np.isnan(data)]
    
    # Create histogram
    n, bin_edges, patches = ax.hist(data, bins=bins, alpha=alpha, 
                                     edgecolor=edgecolor, color=color,
                                     density=density)
    
    # Add statistical markers
    if show_mean:
        mean_val = np.mean(data)
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {mean_val:.4f}')
    
    if show_median:
        median_val = np.median(data)
        ax.axvline(median_val, color='green', linestyle='--', linewidth=2,
                   label=f'Median: {median_val:.4f}')
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    if show_mean or show_median:
        ax.legend(fontsize=10)
    
    plt.tight_layout()
    return fig


def plot_scatter(x: Union[np.ndarray, pd.Series],
                 y: Union[np.ndarray, pd.Series],
                 xlabel: str = 'X',
                 ylabel: str = 'Y',
                 title: str = 'Scatter Plot',
                 figsize: tuple = (10, 6),
                 alpha: float = 0.6,
                 s: float = 20,
                 color: Optional[str] = None,
                 add_regression: bool = False,
                 add_loess: bool = False,
                 loess_frac: float = 0.3,
                 regression_color: str = 'red',
                 loess_color: str = 'orange',
                 show_legend: bool = True):
    """
    Plot a scatter plot with optional regression line and LOESS smoother.
    
    Parameters:
    -----------
    x : np.ndarray or pd.Series
        X-axis data
    y : np.ndarray or pd.Series
        Y-axis data
    xlabel : str, default 'X'
        Label for x-axis
    ylabel : str, default 'Y'
        Label for y-axis
    title : str, default 'Scatter Plot'
        Plot title
    figsize : tuple, default (10, 6)
        Figure size (width, height)
    alpha : float, default 0.6
        Transparency level for points
    s : float, default 20
        Size of points
    color : str, optional
        Color of points (default: matplotlib default)
    add_regression : bool, default False
        Whether to add linear regression line
    add_loess : bool, default False
        Whether to add LOESS smoother
    loess_frac : float, default 0.3
        Fraction of data used for LOESS smoothing (0-1)
    regression_color : str, default 'red'
        Color for regression line
    loess_color : str, default 'orange'
        Color for LOESS line
    show_legend : bool, default True
        Whether to show legend
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Convert to numpy arrays if needed
    if isinstance(x, pd.Series):
        x = x.values
    if isinstance(y, pd.Series):
        y = y.values
    
    # Remove NaN values
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]
    
    # Create scatter plot
    ax.scatter(x_clean, y_clean, alpha=alpha, s=s, color=color, edgecolors='black', 
               linewidth=0.5)
    
    # Add regression line
    if add_regression:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean, y_clean)
        x_line = np.linspace(x_clean.min(), x_clean.max(), 100)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, color=regression_color, linewidth=2, 
                linestyle='--', 
                label=f'Linear fit (R²={r_value**2:.3f})')
    
    # Add LOESS smoother
    if add_loess:
        if not HAS_SCIPY:
            raise ImportError("scipy is required for LOESS smoothing. Install with: pip install scipy")
        
        try:
            # Sort by x for smoother
            sort_idx = np.argsort(x_clean)
            x_sorted = x_clean[sort_idx]
            y_sorted = y_clean[sort_idx]
            
            # Use UnivariateSpline as a simple smoother (alternative to LOESS)
            # For true LOESS, would need statsmodels, but this provides similar functionality
            n_points = len(x_sorted)
            smoothing_factor = int(n_points * (1 - loess_frac))
            if smoothing_factor < n_points:
                spline = UnivariateSpline(x_sorted, y_sorted, s=smoothing_factor)
                x_smooth = np.linspace(x_sorted.min(), x_sorted.max(), 200)
                y_smooth = spline(x_smooth)
                ax.plot(x_smooth, y_smooth, color=loess_color, linewidth=2,
                       label=f'LOESS smoother (frac={loess_frac})')
        except Exception as e:
            import warnings
            warnings.warn(f"Could not add LOESS smoother: {e}")
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    if show_legend and (add_regression or add_loess):
        ax.legend(fontsize=10)
    
    plt.tight_layout()
    return fig


def plot_binned_scatter(x: Union[np.ndarray, pd.Series],
                        y: Union[np.ndarray, pd.Series],
                        bins: int = 10,
                        xlabel: str = 'X Percentile Bin',
                        ylabel: str = 'Mean Y',
                        title: str = 'Mean Y by X Percentile Bin',
                        figsize: tuple = (10, 6),
                        color: str = 'steelblue',
                        edgecolor: str = 'white',
                        alpha: float = 0.8):
    """
    Plot mean of y values within each percentile bin of x values as a bar chart.
    
    Bins x values into percentiles (e.g., deciles) and plots the mean of y
    in each bin. This is useful for analyzing relationships between variables
    when x has a non-uniform distribution.
    
    Parameters:
    -----------
    x : np.ndarray or pd.Series
        X-axis data (will be binned into percentiles)
    y : np.ndarray or pd.Series
        Y-axis data (mean computed per bin)
    bins : int, default 10
        Number of percentile bins (e.g., 10 for deciles)
    xlabel : str, default 'X Percentile Bin'
        Label for x-axis
    ylabel : str, default 'Mean Y'
        Label for y-axis
    title : str, default 'Mean Y by X Percentile Bin'
        Plot title
    figsize : tuple, default (10, 6)
        Figure size (width, height)
    color : str, default 'steelblue'
        Bar color
    edgecolor : str, default 'white'
        Bar edge color
    alpha : float, default 0.8
        Bar transparency
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
        
    Examples:
    --------
    >>> x = np.random.randn(1000)
    >>> y = x + np.random.randn(1000) * 0.5
    >>> fig = plot_binned_scatter(x, y, bins=10)
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Convert to pandas Series for easier handling
    if isinstance(x, np.ndarray):
        x = pd.Series(x)
    if isinstance(y, np.ndarray):
        y = pd.Series(y)
    
    # Ensure same index
    if not x.index.equals(y.index):
        # If different indices, align them
        common_idx = x.index.intersection(y.index)
        x = x.loc[common_idx]
        y = y.loc[common_idx]
    
    # Remove NaN values
    valid_idx = x.dropna().index.intersection(y.dropna().index)
    x_valid = x.loc[valid_idx]
    y_valid = y.loc[valid_idx]
    
    if len(x_valid) == 0:
        raise ValueError("No valid data points after removing NaN values")
    
    # Compute percentile bin edges
    percentiles = np.linspace(0, 100, bins + 1)
    bin_edges = np.percentile(x_valid, percentiles)
    # Ensure right-most items are included
    bin_edges[-1] += 1e-12
    
    # Create bin labels
    bin_labels = [f"P{i+1}" for i in range(bins)]
    
    # Bin the data using pd.cut
    binned = pd.cut(x_valid, bins=bin_edges, labels=bin_labels, include_lowest=True)
    
    # Compute mean of y in each bin
    bin_means = (
        pd.DataFrame({
            "bin": binned,
            "y": y_valid
        })
        .groupby("bin", observed=True)["y"]
        .mean()
    )
    
    # Create color gradient for bars (optional enhancement)
    # Use gradient if color is None, otherwise use single color
    if color is None:
        colors = plt.cm.viridis(np.linspace(0, 1, len(bin_means)))
    else:
        colors = color
    
    # Plot bars
    bars = ax.bar(range(len(bin_means)), bin_means.values, 
                  color=colors, edgecolor=edgecolor, 
                  linewidth=1.5, alpha=alpha)
    
    # Customize appearance
    ax.set_xticks(range(len(bin_means)))
    ax.set_xticklabels(bin_means.index, rotation=0, ha='center')
    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--', axis='y', zorder=0)
    ax.set_axisbelow(True)
    
    # Add zero line if needed
    if bin_means.min() < 0 < bin_means.max():
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, zorder=1)
    
    # Add value labels on bars (optional)
    for i, (bar, val) in enumerate(zip(bars, bin_means.values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}',
                ha='center', va='bottom' if height >= 0 else 'top',
                fontsize=9, fontweight='bold')
    
    # Improve layout
    plt.tight_layout()
    
    return fig




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
