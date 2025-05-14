import numpy as np
from numpy.typing import NDArray
from scipy.optimize import isotonic_regression
from .distances import euclidean_distance
from . import isotonic_error_table, segment
from typing import Tuple


class Unimodal:
    """
    Class used for fitting k unimodal curves to a length n data vector.
    """
    def __init__(
            self,
            penalty : float = 0.0,
            penalty_by_length : int = 0,
            normalize : bool = False
    ):
        """
        Args: 
            penalty (float): Per-segment penalty. Larger values penalize 
                solutions with more segments.

            penalty_by_length (int): If greater than 0, uses a penalty which is set as the average 
                error among segments of a given legnth. NOTE: If this is greater than 0, it will 
                override the normal penalty parameter. Defaults to 0 in which case the 
                standard penalty parameter is used.

            normalize (bool): If True, normalize the errors for each possible fitted segment 
                to the range [0,1].

        Attrs:
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
            est: Size n array of unimodal model regression estimates to the fitted data vector.
            errors: Size k array of sum of squared errors for each individual segment.
        """
        self.penalty = penalty
        self.penalty_by_length = penalty_by_length
        self.normalize = normalize
        self.inflections = None
        self.directions = None
        self.est = None
        self.errors = None
        self.waves = None

    def iso_regression(
        self,
        y : NDArray,
        start_idx : int,
        end_idx : int,
        increasing : bool
    ) -> Tuple[NDArray, float]:
        """
        Fits an isotonic regression model to a segment of the input data. 

        Args:
            y (np.ndarray): Input 1d data vector. 
            start_idx (int): Starting index for the segment
            end_idx (int): Ending index for the segment
            increasing (bool): Boolean value deciding if the isotonic curve should be 
                monotonically increasing (True) or decreasing (False).

        Returns:
            seg_est, error (np.ndarray, float): Fitted, estimate data vector and its 
                sum of squares error with the original data.
        """
        n = y.shape[0]
        assert len(y.shape) == 1, "Input data must be a 1d array."
        assert end_idx > start_idx, "Ending index must be greater than starting index."
        assert start_idx >=0 and start_idx < n
        assert end_idx > 0 and end_idx <= n

        y = y[start_idx:end_idx]
        seg_est = isotonic_regression(y, increasing = increasing).x
        error = euclidean_distance(y, seg_est)
        return seg_est, error

    
    def unimodal_regression(
        self,
        y : NDArray
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
            y (np.ndarray): Size n input data vector. 

        Returns:
            data_est, errors (Tuple[NDArray, NDArray]): Size n array of unimodal model estimates, 
                along with a size k array of sum of squared errors for each segment.
        """
        assert self.inflections is not None, "Inflections must be set before fitting segments."
        assert self.directions is not None, "Directions must be set before fitting segments."
        assert len(self.inflections) > 1, "Inflections must have at least 2 boudnary points."
        assert len(self.inflections) == len(self.directions) + 1, \
        "Inflections and directions must have compatible lengths."
        for i in range(len(self.directions) - 1):
            assert self.directions[i] == 1 - self.directions[i + 1], "Directions must alternate."
            
        errors = []
        est = np.zeros(len(y))
        est[:] = np.nan
        
        for i in range(len(self.inflections) - 1):                
            start_idx = self.inflections[i]
            end_idx = self.inflections[i + 1]
            data_est_segment, error = self.iso_regression(
                y = y,
                start_idx = start_idx,
                end_idx = end_idx,
                increasing = self.directions[i]
            )
            
            errors.append(error)
            est[start_idx:end_idx] = data_est_segment

        self.est = est
        self.errors = errors
    

    def error_tables(self, y : NDArray) -> Tuple[NDArray, NDArray]:
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
        return isotonic_error_table.error_tables(y, self.normalize)
    

    def segment(self, y : NDArray):
        """
        Given an input data vector, find the minimum cost segmentation boundaries, 
        or inflection points, between fitted isotonic curves. 

        Wrapper for the segment.dynamic_unimodal function.

        Args:
            y (np.ndarray[float64]): Size n input data array.
        """
        self.inflections, self.directions = segment.dynamic_unimodal(
            data = y,
            penalty = self.penalty,
            penalty_by_length = self.penalty_by_length,
            normalize = self.normalize
        )

    
    def get_waves(self) -> NDArray:
        """
        Given fitted isotonic inflection points and their corresponding directions, 
        generates a corresponding array of inflection or change points for full unimodal waves. 
        A wave should generally consist of both an increasing and a decreasing 
        isotonic segment (in that order), and for neat input settings we should find that 
        the array returned is simply inflections[::2] (every other segment boundary).

        However, we may also find that the start or end of the data vector is not 
        a complete wave (i.e. starts in a decreasing mode or ends in an increasing one).
        In these cases on the edges of the data vector, we allow a wave to be defined by
        a single isotonic segment. 

        Returns:
            wave_inflections (np.ndarray[int64]): Array of indices for which consecutive
                entries (i, i+1) describe the boundaries of a unimodal wave segmentation
                for the data array. 
        """
        assert len(self.inflections) > 1, "Inflections must have at least 2 boudnary points."
        assert len(self.inflections) == len(self.directions) + 1, \
        "Inflections and directions must have compatible lengths."
        for i in range(len(self.directions) - 1):
            assert self.directions[i] == 1 - self.directions[i + 1], "Directions must alternate."

        n = len(self.inflections)
        start = 0
        end = n

        if self.directions[0] == 0:
            start = 1
        
        if self.directions[-1] == 1:
            end = n - 1

        wave_inflections = self.inflections[start:end:2]

        if self.directions[0] == 0:
            wave_inflections = np.insert(wave_inflections, 0, self.inflections[0])

        if self.directions[-1] == 1:
            wave_inflections = np.append(wave_inflections, self.inflections[-1])

        return np.array(wave_inflections)
    

    def fit(self, y : NDArray):
        """
        Given an input data vector, fits the unimodal model.

        Args:
            y (np.ndarray[float64]): Size n input data array.
        """
        self.segment(y)
        self.unimodal_regression(y)
        self.waves = self.get_waves()