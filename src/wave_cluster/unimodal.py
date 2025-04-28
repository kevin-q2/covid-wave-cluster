import numpy as np
from numpy.typing import NDArray
from scipy.optimize import isotonic_regression
from .utils import euclidean_distance
from . import isotonic_error_table, segment


class Unimodal:
    """
    Tool for fitting unimodal curves to data.
    """
    def __init__(self):
        pass
    

    def fit_segment(
        self,
        data : NDArray,
        start_idx : int,
        end_idx : int,
        increasing : bool
    ):
        """
        Fits an isotonic regression model to a segment of the input data. 

        Args:
            data (np.ndarray): Input 1d data vector. 
            start_idx (int): Starting index for the segment
            end_idx (int): Ending index for the segment
            increasing (bool): Boolean value deciding if the isotonic curve should be 
                monotonically increasing (True) or decreasing (False).

        Returns:
            data_est, error (np.ndarray, float): Fitted, estimate data vector and its 
                sum of squares error with the original data.
        """
        n = data.shape[0]
        assert len(data.shape) == 1, "Input data must be a 1d array."
        assert end_idx > start_idx, "Ending index must be greater than starting index."
        assert start_idx >=0 and start_idx < n
        assert end_idx > 0 and end_idx <= n

        y = data[start_idx:end_idx]
        data_est = isotonic_regression(y, increasing = increasing).x
        error = euclidean_distance(y, data_est)
        return data_est, error

    
    def fit_segments(
        self,
        data : NDArray,
        inflections : NDArray,
        directions : NDArray
    ):
        """
        Given a predetermined list of inflection/segmentation points, 
        fits unimodal curve(s) to the data by alternating between fitting monotonically increasing 
        and monotonically decreasing isotonic regression models to segments of the data. Segment 
        boundaries are determined by the input array of inflection points. 

        NOTE: By default, this will alternate between increasing and decreasing isotonic 
        curves. For example, if inflections = [0,10,20,30,40] for a data vector with length 40, 
        then an increasing curve will be fit from indices 0 to 10, decreasing from 10 to 20, 
        increasing from 20 to 30, and decreasing from 30 to 40.

        Args:
            data (np.ndarray): Size n input data vector. 
            inflections (np.ndarray[int]): Size k + 1 array of indices for inflection points 
                which define the boundaries for k subsegments of the data. Specifically, 
                items i and (i + 1) of the inflections array define the boundary points for 
                a single segment.
            directions (np.ndarray[int]): Size k array of binary values indicating the
                direction of the isotonic curve for each segment. 1 indicates increasing,
                while 0 indicates decreasing. For example, if inflections = [0,10,20,30,40]
                and directions = [1,0,1,0], then the first segment [0,10) is increasing,
                the second segment [10,20) is decreasing, the third segment [20,30) is increasing,
                and the fourth segment [30,40) is decreasing.

        Returns:
            data_est, errors (Tuple[NDArray, NDArray]): Size n array of unimodal model estimates, 
                along with a size k array of sum of squared errors for each segment.
        """
        errors = []
        data_est = np.zeros(len(data))
        data_est[:] = np.nan
        
        for i in range(len(inflections) - 1):                
            start_idx = inflections[i]
            end_idx = inflections[i + 1]
            data_est_segment, error = self.fit_segment(
                data = data,
                start_idx = start_idx,
                end_idx = end_idx,
                increasing = directions[i]
            )
            
            errors.append(error)
            data_est[start_idx:end_idx] = data_est_segment

        return data_est, errors
    

    def error_tables(self, data : NDArray, normalize : bool = False):
        """
        Access to internal functions for computing isotonic error tables. 

        Args:
            data (np.ndarray): Size n input data vector

            normalize (bool): If True, normalize the error tables to the 
                range [0,1].

        Returns:
            increasing_error_table, decreasing_error_table (Tuple[np.ndarray]): (n x n + 1) Arrays with
                entries (i,j) describes the error of fitting an monotonic increasing 
                isotonic regression model to a segment of the data vector 
                indexed by starting at i and ending at j. 

        """
        return isotonic_error_table.error_tables(data, normalize)
    

    def segment(self, data : NDArray, penalty : float = 0.0):
        """
        Given an input data vector, find the minimum cost segmentation boundaries, 
        or inflection points, between fitted isotonic curves. 

        Args:
            data (np.ndarray[float64]): Size n input data array.

            penalty (float): Per-segment penalty. Larger values penalize solutions with more segments.

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
        inflections, directions = segment.dynamic_unimodal(data, penalty)
        return inflections, directions
