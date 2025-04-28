import numpy as np
import pytest
from wave_cluster.isotonic_error_table import (
    increasing_error_table,
    decreasing_error_table,
    error_tables
)

def test_increasing_error_table_basic():
    data = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    table = increasing_error_table(data, normalize=False)
    
    # Assert the shape of the table
    assert table.shape == (len(data), len(data) + 1)

    # Assert NaNs at or below the diagonal
    for i in range(len(data)):
        for j in range(i + 1):
            assert np.isnan(table[i, j])

    # Assert no NaNs in the rest of the table
    for i in range(len(data)):
        for j in range(i + 1, len(data) + 1):
            assert ~np.isnan(table[i, j])
    
    # Assert values 1 off diagonal are 0s
    assert table[0, 1] == 0
    assert table[1, 2] == 0
    assert table[2, 3] == 0
    assert table[3, 4] == 0
    assert table[4, 5] == 0
    
    # Assert specific values for known segments
    assert table[0, 3] == 0  # Error for segment [1, 2, 3]
    assert table[3, 5] > 0  # Error for segment [2, 1]


def test_decreasing_error_table_basic():
    data = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    table = decreasing_error_table(data, normalize=False)
    
    # Assert the shape of the table
    assert table.shape == (len(data), len(data) + 1)
    
    # Assert NaNs at or below the diagonal
    for i in range(len(data)):
        for j in range(i + 1):
            assert np.isnan(table[i, j])

    # Assert no NaNs in the rest of the table
    for i in range(len(data)):
        for j in range(i + 1, len(data) + 1):
            assert ~np.isnan(table[i, j])
    
    # Assert values 1 off diagonal are 0s
    assert table[0, 1] == 0
    assert table[1, 2] == 0
    assert table[2, 3] == 0
    assert table[3, 4] == 0
    assert table[4, 5] == 0
    
    # Assert specific values for known segments
    assert table[0, 3] > 0  # Error for segment [1, 2, 3]
    assert table[3, 5] == 0  # Error for segment [2, 1]


def test_normalization():
    data = np.array([100.0, 200.0, 300.0, 200.0, 100.0])
    table = increasing_error_table(data, normalize=True)
    
    # Assert all non NaN values are in the range [0, 1]
    for i in range(len(data)):
        for j in range(i + 1, len(data) + 1):
            assert 0 <= table[i, j] <= 1


def test_concurrent_normalization():
    data = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    table1, table2 = error_tables(data, normalize=True)
    
    # Assert all non NaN values are in the range [0, 1]
    for i in range(len(data)):
        for j in range(i + 1, len(data) + 1):
            assert 0 <= table1[i, j] <= 1
            assert 0 <= table2[i, j] <= 1


def test_empty_array():
    data = np.array([])
    with pytest.raises(ValueError, match="Input data cannot be empty."):
        increasing_error_table(data)

def test_single_element_array():
    data = np.array([2.0])
    table = increasing_error_table(data, normalize=False)
    
    # Assert the shape of the table
    assert table.shape == (1, 2)
    
    # Assert no error for the single point
    assert np.isnan(table[0, 0])
    assert table[0, 1] == 0

def test_identical_values():
    data = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
    table = increasing_error_table(data, normalize=False)
    table2 = decreasing_error_table(data, normalize=False)
    
    # Assert no error for any segment
    for i in range(len(data)):
        for j in range(i + 1, len(data) + 1):
            assert table[i, j] == 0
            assert table2[i, j] == 0

    # Assert NaNs at or below the diagonal
    for i in range(len(data)):
        for j in range(i + 1):
            assert np.isnan(table[i, j])
            assert np.isnan(table2[i, j])

def test_strictly_increasing():
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    table = increasing_error_table(data, normalize=False)
    
    # Assert no error for any segment
    for i in range(len(data)):
        for j in range(i + 1, len(data) + 1):
            assert table[i, j] == 0

    # Assert NaNs at or below the diagonal
    for i in range(len(data)):
        for j in range(i + 1):
            assert np.isnan(table[i, j])

def test_strictly_decreasing():
    data = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    table = decreasing_error_table(data, normalize=False)
    
    # Assert no error for any segment
    for i in range(len(data)):
        for j in range(i + 1, len(data) + 1):
            assert table[i, j] == 0

    # Assert NaNs at or below the diagonal
    for i in range(len(data)):
        for j in range(i + 1):
            assert np.isnan(table[i, j])
            

def test_multidimensional_array():
    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    with pytest.raises(ValueError):
        increasing_error_table(data)

def test_large_array():
    data = np.random.rand(1000)
    table = increasing_error_table(data, normalize=False)
    
    # Assert the shape of the table
    assert table.shape == (len(data), len(data) + 1)

    # Assert NaNs at or below the diagonal
    for i in range(len(data)):
        for j in range(i + 1):
            assert np.isnan(table[i, j])

    # Assert no NaNs in the rest of the table
    for i in range(len(data)):
        for j in range(i + 1, len(data) + 1):
            assert ~np.isnan(table[i, j])