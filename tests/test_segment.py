import numpy as np
import pytest
from wave_cluster.segment import dynamic_unimodal


# NOTE: The following tests were written with the assistance of generative AI. 
# Nevertheless, they have reviewed, modified, and validated to ensure correctness and reliability.


def test_dynamic_unimodal_basic_case():
    data = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    inflections, directions = dynamic_unimodal(data, c=0.5)
    
    # Assert the inflections and directions are as expected
    assert inflections == [0, 3, 5]
    assert directions == [1, 0]

def test_dynamic_unimodal_empty_array():
    data = np.array([])
    with pytest.raises(ValueError, match="Input data cannot be empty."):
        dynamic_unimodal(data, c=0.5)

def test_dynamic_unimodal_single_element():
    data = np.array([1.0])
    inflections, directions = dynamic_unimodal(data, c=0.5, normalize=False)
    
    # Assert the segmentation handles single-element arrays correctly
    assert inflections == [0, 1]
    assert directions == [1] or directions == [0]

def test_dynamic_unimodal_identical_values():
    data = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
    inflections, directions = dynamic_unimodal(data, c=0.5, normalize=False)
    
    # Assert the segmentation handles identical values correctly
    assert inflections == [0, 5]
    assert directions == [1] or directions == [0]

def test_dynamic_unimodal_strictly_increasing():
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    inflections, directions = dynamic_unimodal(data, c=0.5, normalize=False)
    
    # Assert the segmentation handles strictly increasing data correctly
    assert inflections == [0, 5]
    assert directions == [1]

def test_dynamic_unimodal_strictly_decreasing():
    data = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    inflections, directions = dynamic_unimodal(data, c=0.5, normalize=False)
    
    # Assert the segmentation handles strictly decreasing data correctly
    assert inflections == [0, 5]
    assert directions == [0]

def test_dynamic_unimodal_large_penalty():
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 1.0])
    inflections, directions = dynamic_unimodal(data, c=1.0)
    
    # With a large penalty, the function should prefer fewer segments
    assert inflections == [0, 6]
    assert directions == [1]

def test_dynamic_unimodal_small_penalty():
    data = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    inflections, directions = dynamic_unimodal(data, c=0.01)
    
    # With a small penalty, the function should allow more segments
    assert inflections == [0, 3, 5]
    assert directions == [1, 0]

def test_dynamic_unimodal_no_penalty():
    data = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    inflections, directions = dynamic_unimodal(data, c=0.0)
    
    # With no penalty, the function should allow all segments
    assert inflections == [0, 1, 2, 3, 4, 5]
    assert directions == [1, 0, 1, 0, 1] or directions == [0, 1, 0, 1, 0]

def test_dynamic_unimodal_large_array():
    data = np.random.rand(1000).astype(np.float64)
    inflections, directions = dynamic_unimodal(data, c=0.5)
    
    # Assert the function runs without errors for large arrays
    assert len(inflections) > 0
    assert len(directions) > 0

    # Assert the directions array contains binary values
    assert all(d in [0, 1] for d in directions)

    # Assert the inflections array is alternating
    for i in range(len(directions) - 1):
        assert directions[i] == 1 - directions[i + 1]


def test_dynamic_unimodal_multidimensional_array():
    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    with pytest.raises(ValueError):
        dynamic_unimodal(data, c=0.5)