import numpy as np
from wave_cluster import Unimodal
import pytest


# NOTE: The following tests were written with the assistance of generative AI. 
# Nevertheless, they have been reviewed, modified, and validated to
# ensure correctness and reliability.

def test_fit_segment_basic_increasing():
    unimodal = Unimodal()
    data = np.array([1, 2, 3, 4, 5])
    start_idx = 0
    end_idx = 5
    increasing = True

    data_est, error = unimodal.iso_regression(data, start_idx, end_idx, increasing)

    # Assert the fitted segment matches the input (since it's already increasing)
    assert np.array_equal(data_est, data[start_idx:end_idx])
    assert error == 0  # No error for a perfect fit

def test_fit_segment_basic_decreasing():
    unimodal = Unimodal()
    data = np.array([5, 4, 3, 2, 1])
    start_idx = 0
    end_idx = 5
    increasing = False

    data_est, error = unimodal.iso_regression(data, start_idx, end_idx, increasing)

    # Assert the fitted segment matches the input (since it's already decreasing)
    assert np.array_equal(data_est, data[start_idx:end_idx])
    assert error == 0  # No error for a perfect fit

def test_fit_segment_identical_values():
    unimodal = Unimodal()
    data = np.array([3, 3, 3, 3, 3])
    start_idx = 0
    end_idx = 5
    increasing = True

    data_est, error = unimodal.iso_regression(data, start_idx, end_idx, increasing)

    # Assert the fitted segment matches the input (since all values are identical)
    assert np.array_equal(data_est, data[start_idx:end_idx])
    assert error == 0  # No error for identical values

def test_fit_segment_invalid_input():
    unimodal = Unimodal()
    data = np.array([1, 2, 3, 4, 5])

    with pytest.raises(AssertionError, match="Ending index must be greater than starting index."):
        unimodal.iso_regression(data, start_idx=3, end_idx=2, increasing=True)

    with pytest.raises(AssertionError, match="Input data must be a 1d array."):
        unimodal.iso_regression(data.reshape(-1, 1), start_idx=0, end_idx=5, increasing=True)

def test_fit_segments_basic():
    unimodal = Unimodal()
    data = np.array([1, 2, 3, 2, 1])
    inflections = np.array([0, 3, 5])
    directions = np.array([1, 0])  # Increasing, then decreasing
    unimodal.inflections = inflections
    unimodal.directions = directions
    unimodal.unimodal_regression(data)

    # Assert the fitted data matches the expected isotonic regression
    expected_data = np.array([1, 2, 3, 2, 1])  # Already fits the pattern
    assert np.array_equal(unimodal.est, expected_data)

    # Assert the errors are zero (perfect fit)
    assert np.allclose(unimodal.errors, 0)

def test_fit_segments_invalid_inputs():
    unimodal = Unimodal()
    data = np.array([1, 2, 3, 4, 5])
    inflections = np.array([0, 3, 5])
    directions = np.array([1])  # Mismatched lengths
    unimodal.inflections = inflections
    unimodal.directions = directions

    with pytest.raises(AssertionError, match="Inflections and directions must have compatible lengths."):
        unimodal.unimodal_regression(data)

    directions = np.array([1, 1])
    unimodal.directions = directions
    with pytest.raises(AssertionError, match="Directions must alternate."):
        unimodal.unimodal_regression(data)


def test_fit_segments_single_segment():
    unimodal = Unimodal()
    data = np.array([1, 2, 3, 4, 5])
    inflections = np.array([0, 5])
    directions = np.array([1])  # Single increasing segment
    unimodal.inflections = inflections
    unimodal.directions = directions
    unimodal.unimodal_regression(data)

    # Assert the fitted data matches the input (since it's already increasing)
    assert np.array_equal(unimodal.est, data)
    assert unimodal.errors[0] == 0  # No error for a perfect fit
    

def test_waves_basic_case():
    unimodal = Unimodal()
    inflections = np.array([0, 10, 20, 30, 40])
    directions = np.array([1, 0, 1, 0])  # Alternating increasing and decreasing
    unimodal.inflections = inflections
    unimodal.directions = directions
    wave_inflections = unimodal.get_waves()
    
    # Assert the wave boundaries are correct
    assert np.array_equal(wave_inflections, np.array([0, 20, 40]))

def test_waves_starting_with_decreasing():
    unimodal = Unimodal()
    inflections = np.array([0, 10, 20, 30, 40, 50])
    directions = np.array([0, 1, 0, 1, 0])  # Starts with decreasing
    unimodal.inflections = inflections
    unimodal.directions = directions
    wave_inflections = unimodal.get_waves()
    
    # Assert the wave boundaries include the start
    assert np.array_equal(wave_inflections, np.array([0, 10, 30, 50]))

def test_waves_ending_with_increasing():
    unimodal = Unimodal()
    inflections = np.array([0, 10, 20, 30, 40, 50])
    directions = np.array([1, 0, 1, 0, 1])  # Ends with increasing
    unimodal.inflections = inflections
    unimodal.directions = directions
    wave_inflections = unimodal.get_waves()
    
    # Assert the wave boundaries include the end
    assert np.array_equal(wave_inflections, np.array([0, 20, 40, 50]))

def test_waves_single_increasing_segment():
    unimodal = Unimodal()
    inflections = np.array([0, 10])
    directions = np.array([1])  # Single increasing segment
    unimodal.inflections = inflections
    unimodal.directions = directions
    wave_inflections = unimodal.get_waves()
    
    # Assert the wave boundaries are correct
    assert np.array_equal(wave_inflections, np.array([0, 10]))

def test_waves_single_decreasing_segment():
    unimodal = Unimodal()
    inflections = np.array([0, 10])
    directions = np.array([0])  # Single decreasing segment
    unimodal.inflections = inflections
    unimodal.directions = directions
    wave_inflections = unimodal.get_waves()
    
    # Assert the wave boundaries include the start
    assert np.array_equal(wave_inflections, np.array([0, 10]))
    

def test_waves_non_alternating():
    unimodal = Unimodal()
    inflections = np.array([0, 10, 20, 30])
    directions = np.array([1, 1, 0])  # Non-alternating
    unimodal.inflections = inflections
    unimodal.directions = directions
    with pytest.raises(AssertionError, match="Directions must alternate."):
        wave_inflections = unimodal.get_waves()


def test_waves_incompatible():
    unimodal = Unimodal()
    inflections = np.array([0, 10, 20, 30, 40])
    directions = np.array([1, 1, 0])  # Non-alternating
    unimodal.inflections = inflections
    unimodal.directions = directions
    with pytest.raises(
        AssertionError,
        match="Inflections and directions must have compatible lengths."
    ):
        wave_inflections = unimodal.get_waves()
    