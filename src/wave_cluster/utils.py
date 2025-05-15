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
    assert front + back < X.shape[axis], "Front + back must be less than the length of the axis"
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
    assert t1 >= 0, "Starting index must be non-negative."
    assert t2 >=0, "Ending index must be non-negative."
    assert t1 < len(x), "Starting index must be less than the length of the array."
    assert t2 <= len(x), "Ending index must be less than or equal to the length of the array."
    x_masked = np.zeros(len(x)) + fill
    x_masked[t1:t2] = x[t1:t2]
    return x_masked

        
####################################################################################################


def get_timed_clusters(
        pool : NDArray,
        cluster_labels : NDArray,
        time_idx : int,
        fraction : float = 1.0
) -> NDArray:
    """
    Find all clusters which are active at a given time. 
    A cluster is considered active if the fraction of its members 
    that are active at the given time is greater than or equal to the input fraction.
    More specifically this is designed for the use case of clustering waves.
    We say that a wave is active at time t if start of the wave <= t < end of the wave.
    In other words, t falls within the boundaries of the wave. 

    Args:
        pool (np.ndarray): Pool of waves, where each row is a wave and columns are 
            [location_index, start, end]. 
        cluster_labels (np.ndarray): Labels of the clusters.
        time_idx (int): Time index to check for active clusters.
        fraction (float): Fraction of members in the cluster that must be active at the given time.

    Returns:
        timed_clusters (np.ndarray): 1d array of cluster labels that are active at the given time.
    """
    timed_clusters = []
    unique_labels = np.unique(cluster_labels)
    for c in unique_labels:
        clust = set(np.where(cluster_labels == c)[0])
        satisfies = 0
        for i in clust:
            if pool[i,1] <= time_idx and pool[i,2] > time_idx:
                satisfies += 1

        if satisfies/len(clust) >= fraction:
            timed_clusters.append(c)

    return np.array(timed_clusters)


##########################################################################################


def get_common_segments(
        df : pd.DataFrame,
        pool : NDArray,
        wave_indices : NDArray,
        mask : bool = True
) -> pd.DataFrame:
    """
    Get the common time segments from a set of waves. Specifically, this function 
    takes a dataframe and a list of waves, where each wave is defined by 
    a column index and start/end times. For a given subset of those waves, 
    we then do the following:

    1. Compute the intersection of the start and end times of the waves.
    2. If the intersection is empty, return None.
    3. Otherwise, return a DataFrame containing the common time segments for the waves in the pool.
    4. If mask is True, mask the DataFrame so that only the common time segments are shown.
     
    Args:
        df (pd.DataFrame): DataFrame containing the original set of data for the waves.
        pool (np.ndarray): Pool of waves, where each row is a wave and columns are 
            [location_index, start, end].
        wave_indices (np.ndarray): Indices of the waves to consider in the pool.
        mask (bool): If True, mask the DataFrame so that only the common time segments are shown.
            Defaults to True.

    Returns:
        pd.DataFrame: DataFrame containing the common time segments for the waves in the pool.
            If no common time segments exist, return None.
    """
    pool_subset = pool[wave_indices,:]
    max_start = np.max(pool_subset[:,1])
    min_finish = np.min(pool_subset[:,2])

    if max_start > min_finish:
        return None

    else:
        if mask:
            masked = pd.Series(
                [True if i < max_start or i >= min_finish else False for i in range(len(df))],
                index = df.index
            )
            return df.mask(masked, np.nan).iloc[:, pool_subset[:,0]]
        else:
            return df.iloc[max_start : min_finish, pool_subset[:,0]]
        

###########################################################################################