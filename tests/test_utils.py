import numpy as np
import pandas as pd
import pytest
from wave_cluster.utils import *

# NOTE: The following tests were written with the assistance of generative AI. 
# Nevertheless, they have been reviewed, modified, and validated to
# ensure correctness and reliability.

def test_window_average_1d():
    x = np.array([1, 2, 3, 4, 5])
    result = window_average_1d(x, front=1, back=1)
    expected = np.array([2.0, 3.0, 4.0])  # Sliding window averages: [1,2,3], [2,3,4], [3,4,5]
    np.testing.assert_array_equal(result, expected)

    x = np.array([1, 2, 3, 4, 5])
    result = window_average_1d(x, front=2, back=2)
    expected = np.array([3.0])
    np.testing.assert_array_equal(result, expected)

    x = np.array([1, 2, 3, 4, 5])
    result = window_average_1d(x, front=1, back=2)
    expected = np.array([10/4, 14/4])
    np.testing.assert_array_equal(result, expected)

    x = np.array([1, 2, 3, 4, 5])
    result = window_average_1d(x, front=2, back=1)
    expected = np.array([10/4, 14/4])
    np.testing.assert_array_equal(result, expected)

    x = np.array([1, 2, 3, 4, 5])
    result = window_average_1d(x, front=3, back=1)
    expected = np.array([3.0])
    np.testing.assert_array_equal(result, expected)

    # Edge case: front + back >= len(x)
    with pytest.raises(AssertionError):
        window_average_1d(x, front=2, back=3)

    # Edge case: x is not 1D
    with pytest.raises(AssertionError):
        window_average_1d(np.array([[1, 2], [3, 4]]), front=1, back=1)


def test_window_average_2d():
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    result = window_average_2d(X, front=1, back=1, axis=0)
    expected = np.array([[4.0, 5.0, 6.0]])  # Averaging along columns
    np.testing.assert_array_equal(result, expected)

    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    result = window_average_2d(X, front=1, back=1, axis=1)
    expected = np.array([[2.0], [5.0], [8.0]])  # Averaging along rows
    np.testing.assert_array_equal(result, expected)

    # Rectangular array:
    X = np.array([[1, 2, 3], [4, 5, 6]])
    result = window_average_2d(X, front=1, back=1, axis=1)
    expected = np.array([[2.0], [5.0]])  # Averaging along rows should work
    np.testing.assert_array_equal(result, expected)

    with pytest.raises(AssertionError):
        window_average_2d(X, front=1, back=1, axis=0) # But columns cannot

    # Edge case: front + back >= len(X)
    with pytest.raises(AssertionError):
        window_average_2d(X, front=2, back=2, axis=0)


def test_wave_mask():
    x = np.array([1, 2, 3, 4, 5])
    result = wave_mask(x, t1=1, t2=4)
    expected = np.array([0, 2, 3, 4, 0])
    np.testing.assert_array_equal(result, expected)

    # Edge case: t1 or t2 out of bounds
    with pytest.raises(AssertionError):
        wave_mask(x, t1=-1, t2=6)
    with pytest.raises(AssertionError):
        wave_mask(x, t1=0, t2=6)


def test_get_timed_clusters():
    pool = np.array([[0, 1, 4], [1, 2, 5], [2, 3, 6]])
    cluster_labels = np.array([0, 1, 1])
    result = get_timed_clusters(pool, cluster_labels, time_idx=3, fraction=1.0)
    expected = np.array([0,1])  # Clusters 0,1 are active at time_idx=3
    np.testing.assert_array_equal(result, expected)

    # Testing basic fractions
    pool = np.array([[0, 1, 4], [1, 2, 5], [2, 3, 6]])
    cluster_labels = np.array([0, 1, 1])
    result = get_timed_clusters(pool, cluster_labels, time_idx=2, fraction=1.0)
    expected = np.array([0])
    np.testing.assert_array_equal(result, expected)

    pool = np.array([[0, 1, 4], [1, 2, 5], [2, 3, 6]])
    cluster_labels = np.array([0, 1, 1])
    result = get_timed_clusters(pool, cluster_labels, time_idx=2, fraction=0.5)
    expected = np.array([0,1])
    np.testing.assert_array_equal(result, expected)

    # Edge case: No active clusters
    result = get_timed_clusters(pool, cluster_labels, time_idx=0, fraction=1.0)
    expected = np.array([])
    np.testing.assert_array_equal(result, expected)

    # Edge case: Fraction 0
    result = get_timed_clusters(pool, cluster_labels, time_idx=0, fraction=0.0)
    expected = np.array([0,1])
    np.testing.assert_array_equal(result, expected)



def test_get_common_segments():
    df = pd.DataFrame({
        0: [1, 2, 3, 4, 5],
        1: [6, 7, 8, 9, 10]
    })
    pool = np.array([[0, 1, 4], [1, 2, 5]])
    wave_indices = np.array([0, 1])

    # Case: Common segment exists
    result = get_common_segments(df, pool, wave_indices, mask=False)
    expected = pd.DataFrame({
        0: [3, 4],
        1: [8, 9]
    }, index=[2, 3])
    pd.testing.assert_frame_equal(result, expected)

    # Case: No common segment
    pool = np.array([[0, 1, 2], [1, 3, 5]])
    result = get_common_segments(df, pool, wave_indices, mask=False)
    assert result is None

    # Case: Masked output
    pool = np.array([[0, 1, 4], [1, 2, 5]])
    wave_indices = np.array([0, 1])
    result = get_common_segments(df, pool, wave_indices, mask=True)
    expected = pd.DataFrame({
        0: [np.nan, np.nan, 3, 4, np.nan],
        1: [np.nan, np.nan, 8, 9, np.nan]
    })
    pd.testing.assert_frame_equal(result, expected)


def test_percent_overlap():
    # Test case 1: Fully overlapping segments
    seg1 = (0, 10)
    seg2 = (0, 10)
    assert percent_overlap(seg1, seg2) == 1.0, "Expected 100% overlap for identical segments"

    # Test case 2: Partial overlap
    seg1 = (0, 10)
    seg2 = (5, 15)
    assert percent_overlap(seg1, seg2) == 0.5, "Expected 50% overlap for partial overlap"

    # Test case 3: No overlap
    seg1 = (0, 10)
    seg2 = (10, 20)
    assert percent_overlap(seg1, seg2) == 0.0, "Expected 0% overlap for non-overlapping segments"

    # Test case 4: One segment completely inside the other
    seg1 = (0, 10)
    seg2 = (2, 8)
    assert percent_overlap(seg1, seg2) == 1.0, "Expected 100% overlap for one segment inside the other"

    # Test case 5: Reverse order of segments
    seg1 = (5, 15)
    seg2 = (0, 10)
    assert percent_overlap(seg1, seg2) == 0.5, "Expected 50% overlap for reversed segment order"

    # Test case 6: Invalid input - segment start >= end
    with pytest.raises(ValueError, match="Segment start must be less than segment end."):
        percent_overlap((10, 5), (0, 10))

    # Test case 7: Invalid input - segment length not 2
    with pytest.raises(ValueError, match="Both segments must be tuples or array-like of length 2."):
        percent_overlap((0,), (0, 10))

    # Test case 8: Negative values in segments
    with pytest.raises(ValueError, match="Segment start must be non-negative."):
        seg1 = (-5, 5)
        seg2 = (0, 10)
        percent_overlap(seg1, seg2)

    # Test case 9: Zero-length segment
    seg1 = (0, 0)
    seg2 = (0, 10)
    with pytest.raises(ValueError, match="Segment start must be less than segment end."):
        percent_overlap(seg1, seg2)