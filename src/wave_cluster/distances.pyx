cimport cython
import numpy as np
from sklearn.metrics.pairwise import haversine_distances
cimport numpy as cnp
cnp.import_array()
from libc.math cimport fabs

# Typing
from typing import Tuple, List, Callable, Union
from numpy.typing import NDArray
DTYPE = np.float64
ctypedef cnp.float64_t DTYPE_t
ctypedef cnp.int64_t DTYPE_int_t

# NOTE: This is written in cython mainly to allow for better performance with dtw_distance. 
#       The other functions are not cythonized, but are included here for completeness.

####################################################################################################


def euclidean_distance(
    x : Union[float, int, NDArray],
    y : Union[float, int, NDArray]
) -> float:
    """
    Euclidean Distance between points x and y.

    Args:
        x (np.ndarray): point in euclidean space
        y (np.ndarray): point in euclidean space

    Returns:
        (float): computed distance
    """
    if isinstance(x, float) or isinstance(x, int):
        x = np.array([x])
    if isinstance(y, float) or isinstance(y, int):
        y = np.array([y])

    assert x.shape == y.shape, "x and y must have the same shape."

    return np.linalg.norm(x - y, ord = 2)


####################################################################################################


def dtw_distance(
    cnp.ndarray[DTYPE_t, ndim = 1] x,
    cnp.ndarray[DTYPE_t, ndim = 1] y,
    cnp.ndarray[DTYPE_t, ndim = 1] mult_penalty,
    cnp.ndarray[DTYPE_t, ndim = 1] add_penalty
) -> Tuple[float, List[Tuple[int]]]:
    """
    Computes the dynamic time warp distance between two sequences x and y.
    The dtw distance works by creating a matching or alignment between the 
    two sequences. The distance or cost of the alignment is computed by summing the
    euclidean distances between the aligned elements.

    This implementation allows for the use of different penalties for different types of
    sequence alignment 'moves.' Specifically we consider the following three scenarios 
    when aligning the i-th element of x with the j-th element of y. For each, 
    we can apply a different multiplicative penalty and a different additive penalty to 
    the objective. The dynamic program works by taking the minimum of these three options.

    1. The previous pair in the alignment matched x[i-1] with y[j] (vertical move).
        dtw(i,j) = mult_penalty[0] * distance_fn(x[i], y[j]) + dtw(i-1,j) + add_penalty[0]
    2. The previous pair in the alignment matched x[i] with y[j-1] (horizontal move).
        dtw(i,j) = mult_penalty[1] * distance_fn(x[i], y[j]) + dtw(i,j-1) + add_penalty[1]
    3. The previous pair in the alignment matched x[i] with y[j] (diagonal move).
        dtw(i,j) = mult_penalty[2] * distance_fn(x[i], y[j]) + dtw(i-1,j-1) + add_penalty[2]

    NOTE: That distance_fn is assumed to be the euclidean distance. Since this is only ever 
        computed for pairs of 1d points, its computationally efficient to do so, since 
        we can simply use the absolute difference.

    Args:
        x (np.ndarray[float64]): First sequence.
        y (np.ndarray[float64]): Second sequence.
        mult_penalty (List[float]): List of length 3 which describe multiplicative penalties 
            for vertical, horizontal, and diagonal moves respectively.
        add_penalty (List[float]): List of length 3 which describe additive penalties 
            for vertical, horizontal, and diagonal moves respectively.

    Returns:
        distance (float): The dtw distance between the two sequences.
        alignment (List[Tuple[int]]): A list of tuples describing the alignment between 
            the two sequences. Each tuple is of the form (i,j) indicating
            that x[i] has been matched with y[j].
    """
    # Initialize variables
    cdef int i, j
    cdef int n = len(x)
    cdef int m = len(y)
    cdef cnp.ndarray[DTYPE_t, ndim = 2] cost_array = np.zeros((n,m), dtype=DTYPE)
    cost_array[:] = np.nan
    cdef DTYPE_int_t min_move
    cdef cnp.ndarray[DTYPE_t, ndim=1] costs = np.zeros(3, dtype=DTYPE)
    cdef DTYPE_t dist

    # Compute entries for cost array and track predecessors for alignment
    for i in range(n):
        for j in range(m):
            dist = abs(x[i] - y[j]) # euclidean distance in 1D
            if i == 0 and j == 0:
                cost_array[i,j] = dist
            elif i == 0:
                cost_array[i,j] = (
                    mult_penalty[1] * dist +
                    cost_array[i,j-1] +
                    add_penalty[1]
                )
            elif j == 0:
                cost_array[i,j] = (
                    mult_penalty[0] * dist + 
                    cost_array[i-1,j] + 
                    add_penalty[1]
                )
            else:
                costs[0] = (
                    mult_penalty[0] * dist + cost_array[i-1, j] + add_penalty[0]
                )
                costs[1] = (
                    mult_penalty[1] * dist + cost_array[i, j-1] + add_penalty[1]
                )
                costs[2] = (
                    mult_penalty[2] * dist + cost_array[i-1, j-1] + add_penalty[2]
                )
                
                min_move = np.argmin(costs)

                # Always prefer a diagonal move in the event of ties.
                if costs[min_move] == costs[2]:
                    min_move = 2

                cost_array[i,j] = costs[min_move]
            
    # Backtrack to find the optimal alignment
    i = n - 1 
    j = m - 1
    cdef DTYPE_t current_cost
    alignment = [(i, j)]
    while i > 0 or j > 0:
        current_cost = cost_array[i, j]
        dist = abs(x[i] - y[j]) # euclidean distance in 1D
        vertical_cost = (
            mult_penalty[0] * dist + cost_array[i-1, j] + add_penalty[0]
        )
        horizontal_cost = (
            mult_penalty[1] * dist + cost_array[i, j-1] + add_penalty[1]
        )
        diagonal_cost = (
            mult_penalty[2] * dist + cost_array[i-1, j-1] + add_penalty[2]
        )
        
        if current_cost == diagonal_cost:
            i -= 1
            j -= 1
            alignment = [(i, j)] + alignment
        elif current_cost == vertical_cost:
            i -= 1
            alignment = [(i, j)] + alignment
        else:
            j -= 1
            alignment = [(i, j)] + alignment
        
    return cost_array[n-1,m-1], alignment


####################################################################################################


def haversine(x : NDArray, y : NDArray) -> float:
    """
    Given two locations represented by (latitude, longitude) tuples,
    compute and return the distance in miles between them.

    NOTE: This is mostly designed as a wrapper to
    the sklearn.metrics.pairwise.haversine_distances function, which takes 
    locations in radians instead of latitude/longitiude. 

    Args:
        x (np.ndarray): First location, length two array with [latitude, longitude].
        y (np.ndarray): Second location, length two array with [latitude, longitude].
    Returns:
        (float): Distance in miles between the two locations.
    """
    assert x.shape == (2,) and y.shape == (2,), "x and y must be length 2 arrays."
    x_rad = np.radians(x)
    y_rad = np.radians(y)
    result = haversine_distances([x_rad, y_rad])
    result *= 6371000/1000 * 0.621371  # multiply by Earth radius to get distance in miles
    return result[0,1]


####################################################################################################


def disagreement_distance(P : NDarray[int], Q : NDArray[int]) -> float:
    """
    Compute the disagreement distance between two segmentations of some time series vector x.
    Each segmentation should be represented as an array of indices describing the 
    boundaries between segments. For example, if we have a time series of length 10,
    and we want to segment it into 3 segments, we might have the following segmentation:

    P = [0, 3, 7, 10]. 

    The first segment would be x[0:3], the second segment would be x[3:7], and the
    third segment would be x[7:10]. Any segmentation should start with 0 and 
    end with len(x). The segmentations P and Q need not be the same length. 

    The disagreement distance is then computed by accounting for pairs of indices from x 
    which belong to the same segment in P, but different segments in Q (or vice versa).

    For more information, please see the following reference:
    Taneli Mielikäinen, Evimaria Terzi, and Panayiotis Tsaparas. 2006.
    Aggregating time partitions.
    In Proceedings of the 12th ACM SIGKDD international 
    conference on Knowledge discovery and data mining (KDD '06). 
    Association for Computing Machinery, New York, NY, USA, 347–356.
    https://doi.org/10.1145/1150402.1150442

    Args:
        P (np.ndarray[int]): First segmentation, array of indices describing the boundaries.
        Q (np.ndarray[int]): Second segmentation, array of indices describing the boundaries.

    Returns:
        (float): The disagreement distance between the two segmentations.
    """
    total_disagree = 0
    
    for p in range(len(P) - 1):
        Ep = (P[p+1] - P[p])**2 / 2
        total_disagree += Ep
        
    for q in range(len(Q) - 1):
        Eq = (Q[q+1] - Q[q])**2 / 2
        total_disagree += Eq
        
    U = list(set(P).union(set(Q)))
    U.sort()
    for u in range(len(U) - 1):
        Eu = (U[u+1] - U[u])**2 / 2
        total_disagree -= 2*Eu
        
    return total_disagree