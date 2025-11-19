import numpy as np
from scipy import stats
from typing import Tuple, Union

def yeo_johnson_transform(
    data: Union[np.ndarray, list], 
    lmbda: float = None
) -> Tuple[np.ndarray, float]:
    """
    Perform the Yeo–Johnson transformation on the input data.

    Parameters
    ----------
    data : array-like
        Input array, can be 1D or 2D. Yeo–Johnson is applied column-wise for 2D.
    lmbda : float, optional
        The transformation parameter. If None, it is estimated by maximum likelihood.

    Returns
    -------
    transformed : np.ndarray
        The transformed data.
    lmbda_used : float
        The estimated (or user-provided) lambda parameter.
    """
    data = np.asarray(data)
    if data.ndim == 1:
        if lmbda is None:
            # estimate lambda
            transformed, lmbda_used = stats.yeojohnson(data)
        else:
            transformed = stats.yeojohnson(data, lmbda=lmbda)
            lmbda_used = lmbda
        return transformed, lmbda_used
    elif data.ndim == 2:
        # Apply to columns
        transformed = np.zeros_like(data, dtype=np.float64)
        lmbda_list = []
        for i in range(data.shape[1]):
            if lmbda is None:
                col_trans, col_lmbda = stats.yeojohnson(data[:, i])
            else:
                col_trans = stats.yeojohnson(data[:, i], lmbda=lmbda)
                col_lmbda = lmbda
            transformed[:, i] = col_trans
            lmbda_list.append(col_lmbda)
        return transformed, lmbda_list
    else:
        raise ValueError("Input data must be 1D or 2D array-like.")

def winsorize(
    data: Union[np.ndarray, list],
    min_value: float = None,
    max_value: float = None,
    lower_pct: float = 0.05,
    upper_pct: float = 0.95,
    axis: int = None
) -> np.ndarray:
    """
    Winsorize the input data, either by clipping to min/max values or by percentiles.

    Parameters
    ----------
    data : array-like
        Input data to winsorize.
    min_value : float, optional
        Minimum value to cap the data. Used if specified.
    max_value : float, optional
        Maximum value to cap the data. Used if specified.
    lower_pct : float, optional
        Lower percentile for winsorizing (between 0 and 100).
    upper_pct : float, optional
        Upper percentile for winsorizing (between 0 and 100).
    axis : int, optional
        If data is 2D and using percentiles, axis along which to calculate them.

    Returns
    -------
    winsorized : np.ndarray
        Winsorized version of the data.
    """
    data = np.asarray(data)
    winsorized = np.copy(data)

    if (min_value is not None or max_value is not None):
        # Clip to min/max values
        winsorized = np.clip(winsorized,
                             min_value if min_value is not None else -np.inf,
                             max_value if max_value is not None else np.inf)
    elif (lower_pct is not None or upper_pct is not None):
        # Winsorize by percentiles
        lower_pct_val = lower_pct if lower_pct is not None else 0
        upper_pct_val = upper_pct if upper_pct is not None else 100
        if axis is None:
            # Flatten the array for percentile computation if axis is not specified
            lo = np.percentile(winsorized, lower_pct_val)
            hi = np.percentile(winsorized, upper_pct_val)
            winsorized = np.clip(winsorized, lo, hi)
        else:
            # Percentiles are computed along given axis and must be broadcasted
            # This works for axis=0 or axis=1 for 2D arrays
            lo = np.percentile(winsorized, lower_pct_val, axis=axis, keepdims=True)
            hi = np.percentile(winsorized, upper_pct_val, axis=axis, keepdims=True)
            winsorized = np.clip(winsorized, lo, hi)
    else:
        raise ValueError("Must provide either min_value/max_value or lower_pct/upper_pct.")
    return winsorized

