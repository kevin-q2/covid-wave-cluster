cimport cython
import numpy as np
cimport numpy as cnp
cnp.import_array()
from . import isotonic_error_table
import time

# Typing
from numpy.typing import NDArray
DTYPE = np.float64
ctypedef cnp.float64_t DTYPE_t


def dynamic_unimodal(cnp.ndarray[DTYPE_t, ndim = 1] data, float penalty = 0.0, normalize : bool = True):
    """
    Given an input data vector, find the minimum cost segmentation boundaries 
    between fitted isotonic curves. 

    Args:
        data (np.ndarray[float64]): Size n input data array.

        penalty (float): Per-segment penalty. Larger values penalize solutions with more segments.

        normalize (bool): If True, normalize the error tables so that values fall 
            in the range [0,1]. Default is True.

    Returns:
        inflections (np.ndarray[int64]): Array of indices for which consecutive 
            entries (i, i+1) describe the boundaries of segmentation for the data array. 

        directions (np.ndarray[int64]): Binary array with directional information for 
            each fitted segment, where 1s indicate increasing segments and 0s indicate decreasing.
            For the unimodal model, consecutive segments always alternate between 
            increasing and decreasing isotonic curves. 

            For example, we may want to describe an inflections array [0, 10, 20, 30, 40] in 
            which [0,10) is an increasing segment, [10,20) is decreasing, [20,30) is increasing,
            and [30,40) is decreasing. The directions array would then be [1, 0, 1, 0].
    """
    full_start = time.time()

    start1 = time.time()
    if not data.ndim == 1:
        raise ValueError("Input data must be a 1d array.")

    if len(data) == 0:
        raise ValueError("Input data cannot be empty.")
    cdef int n = data.shape[0]
    cdef cnp.ndarray[DTYPE_t, ndim=2] memo_table = np.full((2, n + 1), np.nan, dtype=DTYPE)

    cdef cnp.ndarray[DTYPE_t, ndim=2] inc_error_table, dec_error_table
    
    inc_error_table, dec_error_table = (
        isotonic_error_table.error_tables(data, normalize = normalize)
    )

    # Fill memo table:
    cdef DTYPE_t error, min_error
    cdef int direction,i,j

    # Base case:
    memo_table[0, 0] = 0
    memo_table[1, 0] = 0

    end1 = time.time()
    print(f"Time taken to initialize and compute error tables: {end1 - start1:.4f} seconds")

    start2 = time.time()
    for i in range(1, n + 1):
        for direction in range(2):
            min_error = np.inf
            for j in range(i):
                if direction == 0:
                    error = dec_error_table[j,i] + memo_table[1, j] + penalty
                else:
                    error = inc_error_table[j,i] + memo_table[0, j] + penalty

                if error < min_error:
                    min_error = error

            memo_table[direction, i] = min_error

    end2 = time.time()
    print(f"Time taken to fill memo table: {end2 - start2:.4f} seconds")

    print(memo_table)

    start3 = time.time()
    # Backtrack:
    cdef int current_direction, current_idx
    cdef DTYPE_t current_error
    current_idx = n
    current_direction = 1 if memo_table[1, current_idx] < memo_table[0, current_idx] else 0
    current_error = memo_table[current_direction, current_idx]

    inflections = [current_idx]
    directions = []
    end3 = time.time()
    print(f"Time taken to initialize backtrack: {end3 - start3:.4f} seconds")

    start4 = time.time()
    while current_idx > 0:
        for j in range(current_idx - 1, -1, -1):
            if current_direction == 0:
                est_error = dec_error_table[j,current_idx] + memo_table[1, j] + penalty
            else:
                est_error = inc_error_table[j,current_idx] + memo_table[0, j] + penalty

            if est_error == current_error:
                inflections = [j] + inflections
                directions = [current_direction] + directions
                current_idx = j
                current_direction = 1 - current_direction
                current_error = memo_table[current_direction, current_idx]
                break 

    end4 = time.time()
    print(f"Time taken to backtrack: {end4 - start4:.4f} seconds")

    full_end = time.time()
    print(f"Total time taken: {full_end - full_start:.4f} seconds")
    return inflections, directions