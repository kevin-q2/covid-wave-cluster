import numpy as np
from numpy.typing import NDArray
from scipy.optimize import isotonic_regression
from .utils import euclidean_distance
from . import isotonic_error_table


def error_tables(y : NDArray):
    """
    Access to internal functions for computing isotonic error tables. 

    Args:
        y (np.ndarray): Size n input data vector

    Returns:
        increasing_error_table, decreasing_error_table (Tuple[np.ndarray]): (n x n + 1) Arrays with
            entries (i,j) describes the error of fitting an monotonic increasing 
            isotonic regression model to a segment of the data vector 
            indexed by starting at i and ending at j. 

    """
    return (
        isotonic_error_table.increasing_error_table(y), 
        isotonic_error_table.decreasing_error_table(y)
    )


class Unimodal:
    """
    Tool for fitting unimodal curves to data.
    """
    def __init__(self):
        pass
    

    def fit_isotonic_segment(
        self,
        y : NDArray,
        start_idx : int,
        end_idx : int,
        increasing : bool
    ):
        """
        Fits an isotonic regression model to a segment of the input data. 

        Args:
            y (np.ndarray): Input 1d data vector. 
            start_idx (int): Starting index for the segment
            end_idx (int): Ending index for the segment
            increasing (bool): Boolean value deciding if the isotonic curve should be 
                monotonically increasing (True) or decreasing (False).

        Returns:
            yhat, error (np.ndarray, float): Fitted, estimate data vector and its 
                sum of squares error with the original data.
        """
        n = y.shape[0]
        assert len(y.shape) == 1, "Input data must be a 1d array."
        assert end_idx > start_idx, "Ending index must be greater than starting index."
        assert start_idx >=0 and start_idx < n
        assert end_idx > 0 and end_idx <= n

        y_ = y[start_idx:end_idx]
        yhat = isotonic_regression(y_, increasing = increasing).x
        error = euclidean_distance(y_, yhat)
        return yhat, error

    
    def fit(
        self,
        y : NDArray,
        inflections : NDArray
    ):
        """
        Fits unimodal curve(s) to the data by alternating between fitting monotonically increasing 
        and monotonically decreasing isotonic regression models to segments of the data. Segment 
        boundaries are determined by the input array of inflection points. 

        NOTE: By default, this will alternate between increasing and decreasing isotonic 
        curves. For example, if inflections = [0,10,20,30,40] for a data vector with length 40, 
        then an increasing curve will be fit from indices 0 to 10, decreasing from 10 to 20, 
        increasing from 20 to 30, and decreasing from 30 to 40.

        Args:
            y (np.ndarray): Size n input data vector. 
            inflections (np.ndarray[int]): Size k + 1 array of indices for inflection points 
                which define the boundaries for k subsegments of the data. Specifically, 
                items i and (i + 1) of the inflections array define the boundary points for 
                a single segment.

        Returns:
            yhat, errors (Tuple[NDArray, NDArray]): Size n array of unimodal model estimates, 
                along with a size k array of sum of squared errors for each segment.
        """
        errors = []
        yhat = np.zeros(len(y))
        yhat[:] = np.nan
        
        for i in range(len(inflections) - 1):
            # Alternate between increasing and decreasing
            if i % 2 == 0:
                increasing = True
            else:
                increasing = False
                
            start_idx = inflections[i]
            end_idx = inflections[i + 1]
            error, yhat_segment = self.fit_isotonic_segment(
                y = y,
                start_idx = start_idx,
                end_idx = end_idx,
                increasing = increasing
            )
            
            errors.append(error)
            yhat[start_idx:end_idx] = yhat_segment

        return yhat, errors