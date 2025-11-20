import numpy as np
import pandas as pd
from typing import List, Tuple, Literal, Optional
from scipy.linalg import cho_factor, cho_solve


def time_series_cv_indices(n_samples: int,
                           n_splits: int,
                           train_window_type: Literal['expanding', 'constant'] = 'expanding',
                           min_train_size: int = None,
                           gap: int = 0) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate time series cross-validation indices for splitting data into train/test sets.
    
    Parameters:
    -----------
    n_samples : int
        Total number of samples in the dataset
    n_splits : int
        Number of cross-validation splits to generate
    train_window_type : {'expanding', 'constant'}, default 'expanding'
        - 'expanding': Train window grows with each split (uses all data up to test start)
        - 'constant': Train window has fixed size (sliding window)
    min_train_size : int, optional
        Minimum size of training set. For 'expanding', this is the initial train size.
        For 'constant', this is the fixed train size. If None, defaults to n_samples // (n_splits + 1)
    gap : int, default 0
        Gap between train and test sets (number of samples to skip)
        
    Returns:
    --------
    splits : List[Tuple[np.ndarray, np.ndarray]]
        List of (train_indices, test_indices) tuples for each CV split.
        Indices can be used to slice arrays/dataframes: data[train_indices], data[test_indices]
        
    Examples:
    --------
    >>> # Expanding window with 5 splits
    >>> splits = time_series_cv_indices(100, n_splits=5, train_window_type='expanding')
    >>> train_idx, test_idx = splits[0]
    >>> train_data = df.iloc[train_idx]
    >>> test_data = df.iloc[test_idx]
    
    >>> # Constant window with fixed train size
    >>> splits = time_series_cv_indices(100, n_splits=5, train_window_type='constant', min_train_size=30)
    """
    if n_samples < 2:
        raise ValueError(f"n_samples must be at least 2, got {n_samples}")
    
    if n_splits < 1:
        raise ValueError(f"n_splits must be at least 1, got {n_splits}")
    
    if gap < 0:
        raise ValueError(f"gap must be non-negative, got {gap}")
    
    # Calculate default minimum train size if not provided
    if min_train_size is None:
        min_train_size = max(1, n_samples // (n_splits + 1))
    
    if min_train_size < 1:
        raise ValueError(f"min_train_size must be at least 1, got {min_train_size}")
    
    # Calculate test size (approximately equal across splits)
    remaining_samples = n_samples - min_train_size
    test_size = max(1, remaining_samples // n_splits)
    
    # Adjust if we have a gap
    available_for_splits = remaining_samples - gap * n_splits
    if available_for_splits < test_size * n_splits:
        # Reduce test size if needed
        test_size = max(1, available_for_splits // n_splits)
    
    splits = []
    
    if train_window_type == 'expanding':
        # Expanding window: train set grows, test set moves forward
        current_train_end = min_train_size
        
        for i in range(n_splits):
            # Calculate test start and end
            test_start = current_train_end + gap
            test_end = min(test_start + test_size, n_samples)
            
            # Check if we have enough data
            if test_start >= n_samples:
                break
            
            # Train indices: from start to current_train_end
            train_indices = np.arange(0, current_train_end)
            
            # Test indices: from test_start to test_end
            test_indices = np.arange(test_start, test_end)
            
            if len(train_indices) > 0 and len(test_indices) > 0:
                splits.append((train_indices, test_indices))
            
            # Move train window end forward for next iteration
            current_train_end = test_end
    
    elif train_window_type == 'constant':
        # Constant window: train set has fixed size, slides forward
        current_train_start = 0
        
        for i in range(n_splits):
            # Train indices: fixed size window
            train_end = current_train_start + min_train_size
            if train_end > n_samples:
                break
            
            train_indices = np.arange(current_train_start, train_end)
            
            # Test indices: after gap
            test_start = train_end + gap
            test_end = min(test_start + test_size, n_samples)
            
            if test_start >= n_samples:
                break
            
            test_indices = np.arange(test_start, test_end)
            
            if len(train_indices) > 0 and len(test_indices) > 0:
                splits.append((train_indices, test_indices))
            
            # Move train window forward
            current_train_start = test_end
    
    else:
        raise ValueError(f"train_window_type must be 'expanding' or 'constant', got {train_window_type}")
    
    if len(splits) == 0:
        raise ValueError(
            f"Could not generate any splits. Try reducing n_splits, "
            f"min_train_size ({min_train_size}), or gap ({gap})"
        )
    
    return splits


def time_series_cv_split(data: pd.DataFrame,
                          n_splits: int,
                          train_window_type: Literal['expanding', 'constant'] = 'expanding',
                          min_train_size: int = None,
                          gap: int = 0) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Split a DataFrame into train/test sets using time series cross-validation.
    
    This is a convenience wrapper around time_series_cv_indices that directly
    returns split DataFrames instead of indices.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame to split (must have time-ordered rows)
    n_splits : int
        Number of cross-validation splits to generate
    train_window_type : {'expanding', 'constant'}, default 'expanding'
        - 'expanding': Train window grows with each split
        - 'constant': Train window has fixed size (sliding window)
    min_train_size : int, optional
        Minimum size of training set. If None, defaults to len(data) // (n_splits + 1)
    gap : int, default 0
        Gap between train and test sets (number of rows to skip)
        
    Returns:
    --------
    splits : List[Tuple[pd.DataFrame, pd.DataFrame]]
        List of (train_df, test_df) tuples for each CV split
        
    Examples:
    --------
    >>> splits = time_series_cv_split(df, n_splits=5, train_window_type='expanding')
    >>> train_df, test_df = splits[0]
    """
    n_samples = len(data)
    indices_splits = time_series_cv_indices(
        n_samples=n_samples,
        n_splits=n_splits,
        train_window_type=train_window_type,
        min_train_size=min_train_size,
        gap=gap
    )
    
    return [(data.iloc[train_idx], data.iloc[test_idx]) 
            for train_idx, test_idx in indices_splits]


def mean_variance_optimization(expected_returns: np.ndarray,
                               covariance: np.ndarray,
                               risk_penalty: float = 1.0,
                               ridge_penalty: float = 0.0) -> np.ndarray:
    """
    Solve the penalized mean-variance optimization problem (long-short).
    
    Maximizes: μ^T w - (λ/2) * w^T Σ w - (α/2) * ||w||^2
    Subject to: sum(w) = 1
    
    Uses Cholesky decomposition via cho_factor/cho_solve for efficient solution.
    
    Parameters:
    -----------
    expected_returns : np.ndarray
        Expected returns vector (n_assets,)
    covariance : np.ndarray
        Covariance matrix (n_assets, n_assets)
    risk_penalty : float, default 1.0
        Risk aversion parameter (λ). Higher values penalize risk more.
    ridge_penalty : float, default 0.0
        L2 regularization parameter (α). Helps stabilize solution.
        
    Returns:
    --------
    weights : np.ndarray
        Optimal portfolio weights (n_assets,) normalized to sum to 1
        
    Examples:
    --------
    >>> returns = np.array([0.1, 0.12, 0.08])
    >>> cov = np.array([[0.04, 0.01, 0.02],
    ...                 [0.01, 0.05, 0.03],
    ...                 [0.02, 0.03, 0.06]])
    >>> weights = mean_variance_optimization(returns, cov, risk_penalty=2.0)
    >>> print(weights)
    """
    expected_returns = np.asarray(expected_returns, dtype=float).flatten()
    covariance = np.asarray(covariance, dtype=float)
    
    n_assets = len(expected_returns)
    
    # Validate inputs
    if covariance.shape != (n_assets, n_assets):
        raise ValueError(
            f"Covariance matrix shape {covariance.shape} does not match "
            f"expected returns length {n_assets}"
        )
    
    if risk_penalty < 0:
        raise ValueError(f"risk_penalty must be non-negative, got {risk_penalty}")
    
    if ridge_penalty < 0:
        raise ValueError(f"ridge_penalty must be non-negative, got {ridge_penalty}")
    
    # Ensure covariance is symmetric
    covariance = (covariance + covariance.T) / 2
    
    # Build the regularized covariance matrix: λΣ + αI
    A = risk_penalty * covariance + ridge_penalty * np.eye(n_assets)
    
    # Ensure positive definiteness by adding small regularization if needed
    # This handles cases where covariance might not be positive definite
    try:
        c, lower = cho_factor(A)
    except np.linalg.LinAlgError:
        # If Cholesky fails, add regularization to diagonal
        A = A + 1e-8 * np.eye(n_assets)
        c, lower = cho_factor(A)
    
    # Solve: (λΣ + αI)w = μ
    # This gives the unconstrained solution
    weights = cho_solve((c, lower), expected_returns)
    
    # Normalize to satisfy sum(w) = 1 constraint
    weights = weights / np.sum(np.abs(weights))
    
    return weights

