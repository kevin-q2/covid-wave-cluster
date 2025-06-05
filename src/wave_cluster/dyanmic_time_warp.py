import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from typing import Callable, List, Tuple
from . import euclidean_distance, dtw_distance


class DynamicTimeWarp:
    """
    Class for computing, managing, and visualizing dynamic time warping (DTW) distances.

    The DTW distance works by creating a matching or alignment between the 
    two sequences. The distance or cost of the alignment is computed by summing the
    distances between the aligned elements.

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
    """
    def __init__(
            self,
            mult_penalty : NDArray = np.array([1.0, 1.0, 1.0], dtype=np.float64),
            add_penalty : NDArray = np.array([0.0, 0.0, 0.0], dtype=np.float64),
            normalize : bool = False
        ):
        """
        Args:
            mult_penalty (NDArray): List of length 3 which describe multiplicative penalties 
                for vertical, horizontal, and diagonal moves respectively.
            add_penalty (NDArray): List of length 3 which describe additive penalties 
                for vertical, horizontal, and diagonal moves respectively.
            normalize (bool): Whether to normalize the input vectors before computing distance. 
                If true, both input vectors are divided by 
                max(max(x), max(y)) to ensure that the distance between individual points 
                is always between 0 and 1. Defaults to False.
        """
        self.mult_penalty = mult_penalty
        self.add_penalty = add_penalty
        self.normalize = normalize
        #self.distance = None
        #self.alignment = None

    def fit(
        self,
        x: NDArray,
        y: NDArray,
    ) -> float:
        """
        Computes the dynamic time warp distance between two sequences x and y.
        Also used to fit the alignment path between the two sequences.
        This method is a wrapper for the dtw_distance function.

        Args:
            x: First time series (numpy array).
            y: Second time series (numpy array).

        Returns:
            distance (float): The dtw distance between the two sequences.

            alignment (List[Tuple[int]]): A list of tuples describing the alignment between 
            the two sequences. Each tuple is of the form (i,j) indicating
            that x[i] has been matched with y[j].
        """
        if self.normalize:
            norm = max(x.max(), y.max())
            if norm != 0:
                x = x / norm
                y = y / norm

        distance, alignment = dtw_distance(
            x,
            y,
            self.mult_penalty,
            self.add_penalty
        )

        #self.alignment = alignment
        #self.distance = distance
        return distance, alignment
    
    def plot_permutation(self, alignment : List[Tuple[int]], axis : Callable = None):
        """
        Plots the alignment path of the DTW distance.
        Args:
            alignment (List[Tuple[int]]): A list of tuples describing the alignment between 
            the two sequences. Each tuple is of the form (i,j) indicating
            that x[i] has been matched with y[j].

            axis (matplotlib axis): Axis to plot on.
        """
        if axis is None:
            fig,axis = plt.subplots(1,1)

        xs = [i[0] for i in alignment]
        ys = [i[1] for i in alignment]
        axis.plot(xs,ys)


    def plot_alignment(
            self,
            alignment : List[Tuple[int]],
            x : NDArray,
            y : NDArray,
            offset : float = 1,
            skips : int = 2,
            axis : Callable = None
        ):
        """
        Plots the two time series and a visualization for their matched alignment.

        Args:
            alignment (List[Tuple[int]]): A list of tuples describing the alignment between 
                the two sequences. Each tuple is of the form (i,j) indicating
                that x[i] has been matched with y[j].
            x (NDArray): First time series (numpy array).
            y (NDArray): Second time series (numpy array).
            offset (float): Vertical distance between time series for visualization.
            skips (int): Number of indices between consecutive, visualized alignment pairs.
            axis (matplotlib axis): Axis to plot on.
        """
        if axis is None:
            fig,axis = plt.subplots(1,1)

        y_off = y + offset

        axis.plot(x)
        axis.plot(y_off)

        for i in alignment[::skips]:
            axis.plot(
                [i[0],i[1]],
                [x[i[0]], y_off[i[1]]],
                color = 'black',
                alpha = 0.4,
                linestyle = 'dashed',
                linewidth = 0.75
            )