cimport cython
import numpy as np
cimport numpy as cnp
cnp.import_array()
from scipy.optimize import isotonic_regression
from .distances import euclidean_distance

# Typing
from numpy.typing import NDArray
from typing import Tuple
DTYPE = np.float64
ctypedef cnp.float64_t DTYPE_t

# Might be able to speed this up more by using a minimum segement size!

def increasing_error_table(
    cnp.ndarray[DTYPE_t, ndim = 1] data,
    normalize : bool = False
) -> cnp.ndarray[DTYPE_t]:
    """
    Computes a table where each entry (i,j) describes the error of fitting an 
    monotonic increasing isotonic regression model to a segment of a data vector 
    indexed by starting at i and ending at j. 

    Args:
        data (np.ndarray[float64]): Size n input data array.

        normalize (bool): If True, normalize the resulting error table so that values fall 
            in the range [0,1]. Default is False.

    Returns:
        table (np.ndarray[float64]): Size n x (n + 1) error table. Includes an extra 
            column to allow ending index to include the final data entry.
    """
    if not data.ndim == 1:
        raise ValueError("Input data must be a 1d array.")
    if len(data) == 0:
        raise ValueError("Input data cannot be empty.")
    cdef int n = data.shape[0]
    cdef int n_ = n + 1
    cdef cnp.ndarray[DTYPE_t, ndim=2] table = np.full((n, n_), np.nan, dtype=DTYPE)

    cdef cnp.ndarray[DTYPE_t, ndim=1] y
    cdef cnp.ndarray[DTYPE_t, ndim=1] yhat
    cdef DTYPE_t error
    for i in range(n):
        for j in range(i + 1, n + 1):
            y = data[i:j]
            yhat = isotonic_regression(y, increasing = True).x
            error = euclidean_distance(y,yhat)
            table[i,j] = error

    if normalize:
        # Normalize the error table
        max_error = np.nanmax(table)
        if max_error > 0:
            table /= max_error
        else:
            raise ValueError("Maximum error is zero, cannot normalize.")

    return table


def decreasing_error_table(
    cnp.ndarray[DTYPE_t, ndim = 1] data,
    normalize : bool = False
) -> cnp.ndarray[DTYPE_t]:
    """
    Computes a table where each entry (i,j) describes the error of fitting an 
    monotonic increasing isotonic regression model to a segment of a data vector 
    indexed by starting at i and ending at j. 

    Args:
        data (np.ndarray[float64]): Size n input data array.

    Returns:
        table (np.ndarray[float64]): Size n x (n + 1) error table. Includes an extra 
            column to allow ending index to include the final data entry.
    """
    if not data.ndim == 1:
        raise ValueError("Input data must be a 1d array.")
    cdef int n = data.shape[0]
    cdef int n_ = n + 1
    cdef cnp.ndarray[DTYPE_t, ndim=2] table = np.full((n, n_), np.nan, dtype=DTYPE)

    cdef cnp.ndarray[DTYPE_t, ndim=1] y
    cdef cnp.ndarray[DTYPE_t, ndim=1] yhat
    cdef DTYPE_t error
    cdef int i,j
    for i in range(n):
        for j in range(i + 1, n + 1):
            y = data[i:j]
            yhat = isotonic_regression(y, increasing = False).x
            error = euclidean_distance(y,yhat)
            table[i,j] = error

    if normalize: 
        # Normalize the error table
        max_error = np.nanmax(table)
        if max_error > 0:
            table /= max_error
        else:
            raise ValueError("Maximum error is zero, cannot normalize.")
    
    return table


def error_tables(
    cnp.ndarray[DTYPE_t, ndim = 1] data,
    normalize : bool = False
) -> Tuple[
    cnp.ndarray[DTYPE_t],
    cnp.ndarray[DTYPE_t]
]:
    """
    Computes the increasing and decreasing error tables for a given data vector.

    Args:
        data (np.ndarray[float64]): Size n input data array.

        normalize (bool): If True, normalize the resulting error tables so that values fall 
            in the range [0,1]. This function does so by taking the maximum between the 
            two tables, so that their values may be used concurrently. Default is False.

    Returns:
        inc_table (np.ndarray[float64]): Size n x (n + 1) increasing error table. 
            Includes an extra column to allow ending index to include the final data entry.

        dec_table (np.ndarray[float64]): Size n x (n + 1) decreasing error table. 
            Includes an extra column to allow ending index to include the final data entry.
    """
    inc_table = increasing_error_table(data)
    dec_table = decreasing_error_table(data)

    if normalize:
        # Normalize the error tables
        max_error = np.nanmax(np.concatenate((inc_table, dec_table), axis=None))
        if max_error > 0:
            inc_table /= max_error
            dec_table /= max_error
        else:
            raise ValueError("Maximum error is zero, cannot normalize.")

    return inc_table, dec_table


