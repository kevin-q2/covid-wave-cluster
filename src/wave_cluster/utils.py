import numpy as np
import pandas as pd
import math
from itertools import combinations
from scipy.sparse import csr_matrix
from numpy.typing import NDArray


####################################################################################################


def window_average_1d(x : NDArray, front : int, back : int) -> NDArray:
    """
    Take a sliding windown average of a 1D array x using
    front (coming beforehand) elements + back (coming afterwards) elements.

    Args:
        x (np.ndarray): Size n array to be averaged.
        front (int): Number of elements before the current index to include in the average.
        back (int): Number of elements after the current index to include in the average.
    Returns:
        window_avg (np.ndarray): Window averaged array of size n - front - back.
    """
    assert front >= 0, "Front must be non-negative"
    assert back >= 0, "Back must be non-negative"
    assert front + back < x.shape[0], "Front + back must be less than the length of x"
    assert len(x.shape) == 1, "x must be a 1D array"

    n = x.shape[0]
    window_size = front + 1 + back
    window_idx = window_size
    window_avg = np.zeros(n - front - back)
    while window_idx <= n:
        window = x[window_idx - window_size: window_idx]
        window_avg[window_idx - window_size] = np.mean(window)
        window_idx += 1

    return window_avg


####################################################################################################


def window_average_2d(X : NDArray, front : NDArray, back : NDArray, axis : int = 0) -> NDArray:
    """
    Take a sliding windown average of a 2D array X, by computing a windowed 
    average over each column or row.

    Args:
        X (np.ndarray): Size n x n array to be averaged.
        front (int): Number of elements before the current index to include in the average.
        back (int): Number of elements after the current index to include in the average.
        axis (int): Axis along which to compute the windowed average. Default is 0, 
            in which case the windowed average is computed along each column. Alternatively, 
            axis = 1 computes the windowed average along each row.
    Returns:
        window_avg (np.ndarray): Window averaged array of size (n - front - back) x n. 
    """
    assert front >= 0, "Front must be non-negative"
    assert back >= 0, "Back must be non-negative"
    assert front + back < X.shape[0], "Front + back must be less than the length of x"
    assert len(X.shape) == 2, "X must be a 2D array"

    window_avg = np.apply_along_axis(
        window_average_1d,
        axis = axis,
        arr = X,
        front = front,
        back = back
    )
    return window_avg


####################################################################################################


def wave_mask(x : NDArray, t1 : int, t2 : int, fill : float = 0.0) -> NDArray:
    """
    Mask a 1D array x with zeros outside of the range x[t1:t2].
    For example, an array x = [1,2,3,4,5] with t1 = 1 and t2 = 4 would 
    return [0,2,3,4,0]. 

    Args:
        x (np.ndarray): Size n array to be masked.
        t1 (int): Starting index of the mask.
        t2 (int): Ending index of the mask.
        fill (float): Value to fill outside of the mask.

    Returns:
        x_masked (np.ndarray): Masked array of size n.
    """
    x_masked = np.zeros(len(x)) + fill
    x_masked[t1:t2] = x[t1:t2]
    return x_masked

        
####################################################################################################