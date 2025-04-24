cimport cython
import numpy as np
cimport numpy as cnp
cnp.import_array()
from scipy.optimize import isotonic_regression
from .utils import euclidean_distance

# Typing
from numpy.typing import NDArray
DTYPE = np.float64
ctypedef cnp.float64_t DTYPE_t

# Might be able to speed this up more by using a minimum segement size!

def increasing_error_table(cnp.ndarray[DTYPE_t, ndim = 1] data):
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
    for i in range(n):
        for j in range(i + 1, n + 1):
            y = data[i:j]
            yhat = isotonic_regression(y, increasing = True).x
            error = euclidean_distance(y,yhat)
            table[i,j] = error

    return table


def decreasing_error_table(cnp.ndarray[DTYPE_t, ndim = 1] data):
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
    for i in range(n):
        for j in range(i + 1, n + 1):
            y = data[i:j]
            yhat = isotonic_regression(y, increasing = False).x
            error = euclidean_distance(y,yhat)
            table[i,j] = error

    return table


