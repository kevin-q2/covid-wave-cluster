import numpy as np
import copy
from joblib import Parallel, delayed, parallel_config
from typing import Dict, Callable, Any, List, Tuple
from numpy.typing import NDArray

from .unimodal import Unimodal
from .dyanmic_time_warp import DynamicTimeWarp
from . import euclidean_distance
from .utils import wave_mask


class WavePool:
    """
    Class for managing wave segmentation and distance computation for an input dataset. 
    """
    def __init__(
            self,
            wave_module : Callable = None,
            distance_module : Callable = None,
            fit_waves_ : bool = True,
            fit_distances_ : bool = True,
            mask : bool = True,
            threshold : float = np.inf,
            cpu_count : int = 1
        ):
        """
        Args:
            wave_module (Class Object): Class object used to fit unimodal waves to the data.
                    Must have a .fit(x) method which takes a 1d array x and fits the waves. Must 
                    also have a .waves attribute which contains an array describing the boundaries
                    for the fitted waves. See the Unimodal class in unimodal.py for an example.
                    Defaults to None.

            distance_module (Class Object): Class object used to compute the distance 
                between two waves. Must have a .fit(x, y) method which takes two 1d arrays
                and returns the numerical distance between them. 
                See the DynamicTimeWarp class in dyanmic_time_warp.py for an example.
                Defaults to None.

            fit_waves_ (bool): If True, calling .fit() will fit waves to the input data.
                Defaults to True.

            fit_distances_ (bool): If True, calling .fit() will fit distances to the 
                waves described by self.pool. Defaults to True.

            mask (bool): If True, waves will be masked so that while they remain the
                same length as their parent data vector, they are given 0s for time indices 
                outside of the wave segment. This is useful for computing distances between
                waves. Defaults to True.

            threshold (int): The maximum number of days a pair of waves can differ in their 
                start time, before the distance is automatically set to infinity. Defaults to 
                np.inf, in which case distances are computed between all possible pairs of waves. 

            cpu_count (int): Number of processors available to use in parallel.

        Attrs:
            X (np.ndarray): Input dataset of shape (n, m) where n is the number of timesteps
                and m is the number of locations.
            pool (np.ndarray): Array of shape (q, 3) where q is the total number of 
                wave segments pooled from a segmentation across all locations in the data. 
                Each row contains the location index (column of X), start index, 
                and end index for each segment.
            q (int): Total number of wave segments pooled from a segmentation across all
                locations in the fitted data.
            distances (np.ndarray): Array of shape (q, q) containing pairwise distances 
                between all wave segments.
        """
        self.wave_module = wave_module
        self.distance_module = distance_module
        self.fit_waves_ = fit_waves_
        self.fit_distances_ = fit_distances_
        self.mask = mask
        self.cpu_count = cpu_count
        self.X = None
        self.pool = None
        self.q = None
        self.distances = None


    def fit_waves_to_location(
            self,
            loc_idx : int
        ) -> NDArray:
        """
        Fits unimodal waves to a single location. 

        Args:
            loc_idx (int): Location identifier. Column index of the location in the input dataset.
            
        Returns:
            location_pool (np.ndarray): Array of shape (q, 3) where q is the number of 
                wave segments found for the given location. 
                Each row contains the location index, start index, and end index for each segment.
        """
        # Copies the wave module to avoid modifying the original instance.
        # This allows the method to be run in parallel. 
        wave_module = copy.deepcopy(self.wave_module)

        if self.X is None:
            raise ValueError("Input data not fitted. Please run .fit() first.")
        
        x = self.X[:, loc_idx]
        wave_module.fit(x)
        wave_inflections = wave_module.waves

        location_pool = []
        for j in range(len(wave_inflections) - 1):
            start_idx = wave_inflections[j]
            end_idx = wave_inflections[j + 1]
            new_wave = [[loc_idx, start_idx, end_idx]]
            location_pool = location_pool + new_wave

        return np.array(location_pool)
    

    def fit_waves(self):
        """
        Fits unimodal waves to all locations in the input dataset, in parallel.
        """
        if self.X is None:
            raise ValueError("Input data not fitted. Please run .fit() first.")

        n,m = self.X.shape
        location_results = Parallel(n_jobs = self.cpu_count, backend = 'loky')(
            delayed(self.fit_waves_to_location)(loc)
            for loc in range(m)
        )
        
        self.pool = np.vstack(location_results)
        self.q = self.pool.shape[0]


    def fit_distance_pairwise(
            self,
            wave_idx1 : NDArray,
            wave_idx2 : NDArray
        ) -> float:
        """
        Computes the pairwise distance between two waves from the input dataset
        Args:
            wave_idx1 (np.ndarray): Row index to self.pool describing the first wave segment.
            wave_idx2 (np.ndarray): Row index to self.pool describing the second wave segment.
        Returns:
            distance (float): The distance between the two wave segments.
        """
        if self.X is None:
            raise ValueError("Input data not fitted. Please run .fit() first.")
        if self.pool is None:
            raise ValueError("Wave pool is not fitted. Please run .fit_waves() first.")
        
        start1 = self.pool[wave_idx1][1]
        start2 = self.pool[wave_idx2][1]
        if np.abs(start1 - start2) > self.threshold:
            distance = np.inf
        else:
            if self.mask:
                x = wave_mask(
                    self.X[: , self.pool[wave_idx1][0]],
                    self.pool[wave_idx1][1],
                    self.pool[wave_idx1][2]
                )
                y = wave_mask(
                    self.X[: , self.pool[wave_idx2][0]],
                    self.pool[wave_idx2][1],
                    self.pool[wave_idx2][2]
                )
            else:
                x = self.X[self.pool[wave_idx1][1] : self.pool[wave_idx1][2], self.pool[wave_idx1][0]]
                y = self.X[self.pool[wave_idx2][1] : self.pool[wave_idx2][2], self.pool[wave_idx2][0]]

            distance_mod = copy.deepcopy(self.distance_module)
            distance = distance_mod.fit(x, y)

        return distance
        
    
    def fit_distances(self):
        """
        Computes the pairwise distance between all waves found in the input dataset.
        """
        if self.X is None:
            raise ValueError("Input data not fitted. Please run .fit() first.")
        if self.pool is None:
            raise ValueError("Wave pool is not fitted. Please run .fit_waves() first.")
        
        distances = np.zeros((self.q,self.q))

        wave_pair_results = Parallel(n_jobs = self.cpu_count, backend = 'loky')(
            delayed(self.fit_distance_pairwise)(i,j)
            for i in range(self.q) for j in range(i + 1, self.q)
        )
            
        for i in range(self.q):
            for j in range(i + 1, self.q):
                distances[i,j] = wave_pair_results.pop(0)
                distances[j,i] = distances[i,j]

        self.distances = distances


    def fit(self, X : NDArray):
        """
        Fits the input dataset to the wave pool.

        Args:
            X (np.ndarray): Input dataset of shape (n, m) where n is the number of timesteps
                and m is the number of locations.
        """
        assert len(X.shape) == 2, "Input data must be a 2D array."
        assert X.shape[0] > 0, "Input data must have at least one row."
        assert X.shape[1] > 0, "Input data must have at least one column."
        self.X = X

        # Fit waves.
        if self.wave_module is not None and self.fit_waves_:
            self.fit_waves()
        elif self.fit_waves_ and self.wave_module is None:
            raise ValueError("Wave module must be set to fit waves.")
        
        # Fit distances.
        if self.distance_module is not None and self.fit_distances_:
            self.fit_distances()
        elif self.fit_distances_ and self.distance_module is None:
            raise ValueError("Distance module must be set to fit distances.")
            

    def save_pool(self, fname : str):
        """
        Saves the wave pool to a file.

        Args:
            fname (str): Filename to save the wave pool.
        """
        assert fname.endswith('.npz'), "Filename must end with .npz"
        np.savez_compressed(fname, self.pool)
        

    def load_pool(self, fname : str):
        """
        Loads the wave pool from a file.

        Args:
            fname (str): Filename to load the wave pool.
        """
        pool_load = np.load(fname)
        self.pool = pool_load['arr_0']
        self.q = self.pool.shape[0]


    def save_distances(self, fname : str):
        """
        Saves the pairwise distances to a file.

        Args:
            fname (str): Filename to save the pairwise distances.
        """
        assert fname.endswith('.npz'), "Filename must end with .npz"
        np.savez_compressed(fname, self.distances)


    def load_distances(self, fname : str):
        """
        Loads the pairwise distances from a file.

        Args:
            fname (str): Filename to load the pairwise distances.
        """
        distances_load = np.load(fname)
        self.distances = distances_load['arr_0']