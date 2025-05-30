import numpy as np
import pytest
from wave_cluster.distances import dtw_distance, euclidean_distance, haversine, disagreement_distance

# NOTE: The following tests were written with the assistance of generative AI. 
# Nevertheless, they have been reviewed, modified, and validated to
# ensure correctness and reliability.

# -------------------- Tests for dtw_distance --------------------

def test_dtw_distance_basic_case():
    x = np.array([1, 2, 3], dtype=float)
    y = np.array([1, 2, 3], dtype=float)
    mult_penalty = np.array([1.0, 1.0, 1.0])
    add_penalty = np.array([0.0, 0.0, 0.0])
    distance, alignment = dtw_distance(x, y, mult_penalty=mult_penalty, add_penalty=add_penalty)
    
    # Assert the distance is zero for identical sequences
    assert distance == 0
    # Assert the alignment is a perfect diagonal
    assert alignment == [(0, 0), (1, 1), (2, 2)]

def test_dtw_distance_basic_misaligned():
    x = np.array([1, 1, 2, 3, 3], dtype=float)
    y = np.array([1, 2, 2, 2, 3], dtype=float)
    mult_penalty = np.array([1.0, 1.0, 1.0])
    add_penalty = np.array([0.0, 0.0, 0.0])
    distance, alignment = dtw_distance(x, y, mult_penalty=mult_penalty, add_penalty=add_penalty)
    
    assert distance == 0
    assert alignment == [(0, 0), (1, 0), (2, 1), (2, 2), (2, 3), (3, 4), (4, 4)]

def test_dtw_distance_different_lengths():
    x = np.array([1, 2, 3, 3], dtype = float)
    y = np.array([1, 2], dtype = float)
    mult_penalty = np.array([1.0, 1.0, 1.0])
    add_penalty = np.array([0.0, 0.0, 0.0])
    distance, alignment = dtw_distance(x, y, mult_penalty=mult_penalty, add_penalty=add_penalty)
    
    # Assert first element is matched with first element 
    assert alignment[0] == (0, 0)  # First element of x matches first element of y
    # Assert last element is matched with last element
    assert alignment[-1] == (3, 1)  # Last element of x matches last element of y

def test_dtw_distance_custom_penalties():
    x = np.array([1, 2, 3], dtype = float)
    y = np.array([1, 2, 3], dtype = float)
    mult_penalty = np.array([2.0, 2.0, 1.0])
    add_penalty = np.array([1.0, 1.0, 0.0])
    distance, alignment = dtw_distance(x, y, mult_penalty=mult_penalty, add_penalty=add_penalty)
    
    # Assert the distance is still zero for identical sequences
    assert distance == 0
    # Assert the alignment is a perfect diagonal
    assert alignment == [(0, 0), (1, 1), (2, 2)]

    # Multiplicative penalties on their own won't change distance, since 
    # distances between individual elements are zeros.
    x = np.array([1, 1, 2, 3, 3], dtype=float)
    y = np.array([1, 2, 2, 2, 3], dtype=float)
    mult_penalty = np.array([5.0, 5.0, 1.0])
    add_penalty = np.array([0.0, 0.0, 0.0])
    distance, alignment = dtw_distance(x, y, mult_penalty=mult_penalty, add_penalty=add_penalty)
    assert distance == 0
    assert alignment == [(0, 0), (1, 0), (2, 1), (2, 2), (2, 3), (3, 4), (4, 4)]

    # But additive penalties will!
    x = np.array([1, 1, 2, 3, 3], dtype=float)
    y = np.array([1, 2, 2, 2, 3], dtype=float)
    mult_penalty = np.array([1.0, 1.0, 1.0])
    add_penalty = np.array([3.0, 3.0, 0.0])
    distance, alignment = dtw_distance(x, y, mult_penalty=mult_penalty, add_penalty=add_penalty)
    assert distance == 2
    assert alignment == [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]

'''
# Deprecated functionality
def test_dtw_distance_custom_distance_function():
    def manhattan_distance(a, b):
        return abs(a - b)
    
    x = np.array([1, 2, 4], dtype = float)
    y = np.array([1, 2, 3], dtype = float)
    distance, alignment = dtw_distance(x, y, distance_fn=manhattan_distance)
    
    # Assert the distance is zero for identical sequences
    assert distance == 1
    # Assert the alignment is a perfect diagonal
    assert alignment == [(0, 0), (1, 1), (2, 2)]
'''

def test_dtw_distance_single_element():
    x = np.array([1], dtype = float)
    y = np.array([1], dtype = float)
    mult_penalty = np.array([1.0, 1.0, 1.0])
    add_penalty = np.array([0.0, 0.0, 0.0])
    distance, alignment = dtw_distance(x, y, mult_penalty=mult_penalty, add_penalty=add_penalty)
    
    # Assert the distance is zero for identical single-element sequences
    assert distance == 0
    # Assert the alignment is trivial
    assert alignment == [(0, 0)]

def test_dtw_distance_invalid_inputs():
    x = np.array([1, 2, 3], dtype = float)
    y = np.array([[1, 2], [3, 4]], dtype = float)  # Multidimensional array
    mult_penalty = np.array([1.0, 1.0, 1.0])
    add_penalty = np.array([0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match=".*"):
        dtw_distance(x, y, mult_penalty=mult_penalty, add_penalty=add_penalty)

# -------------------- Tests for euclidean_distance --------------------

def test_euclidean_distance_scalars():
    x = 3
    y = 4
    distance = euclidean_distance(x, y)
    assert distance == 1  # |3 - 4|

def test_euclidean_distance_arrays():
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    distance = euclidean_distance(x, y)
    assert np.isclose(distance, 5.196, atol=1e-3)  # sqrt((4-1)^2 + (5-2)^2 + (6-3)^2)

def test_euclidean_distance_invalid_shapes():
    x = np.array([1, 2, 3])
    y = np.array([[4, 5], [6, 7]])  # Multidimensional array
    with pytest.raises(AssertionError, match="x and y must have the same shape."):
        euclidean_distance(x, y)

# -------------------- Tests for haversine --------------------

def test_haversine_basic_case():
    x = np.array([-34.83333, -58.5166646])  # Ezeiza Airport (Buenos Aires, Argentina)
    y = np.array([49.0083899664, 2.53844117956])  # Charles de Gaulle Airport (Paris, France)
    distance = haversine(x, y)
    assert np.isclose(distance, 6896.934624182275, atol=1e-5)  # Approximate distance in miles

def test_haversine_invalid_shapes():
    x = np.array([37.7749, -122.4194])
    y = np.array([34.0522])  # Invalid shape
    with pytest.raises(AssertionError, match="x and y must be length 2 arrays."):
        haversine(x, y)

# -------------------- Tests for disagreement_distance --------------------

def test_disagreement_distance_basic_case():
    P = np.array([0, 3, 7, 10])
    Q = np.array([0, 5, 10])
    distance = disagreement_distance(P, Q)
    
    # Assert the disagreement distance is computed correctly
    assert distance > 0

def test_disagreement_distance_identical_segmentations():
    P = np.array([0, 3, 7, 10])
    Q = np.array([0, 3, 7, 10])
    distance = disagreement_distance(P, Q)
    
    # Assert the disagreement distance is zero for identical segmentations
    assert distance == 0