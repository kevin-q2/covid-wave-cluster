import pytest
import numpy as np

@pytest.fixture
def alternating():
    return np.array([0,1,0,1,0])

@pytest.fixture
def binary_unimodal():
    return np.array(
        [0,0,0,1,1,1,1,0,0,0,1,1,1,1,0,1,1,1,0,0,0,0]
    )