import numpy as np
from wave_cluster import Unimodal, error_tables


def test_isotonic(alternating, binary_unimodal):
    y = alternating
    yhat, error = Unimodal().fit_isotonic_segment(
        y,
        start_idx = 0,
        end_idx = 5,
        increasing = True
    )

    assert error == 1
    assert np.array_equal(yhat, np.array([0,0.5,0.5,0.5,0.5]))

    y = binary_unimodal
    yhat, error = Unimodal().fit_isotonic_segment(
        y,
        start_idx = 0,
        end_idx = 6,
        increasing = True
    )

    assert error == 0
    assert np.array_equal(yhat, y[0:6])


# [0,0,0,1,1,1,1,0,0,0,1,1,1,1,0,1,1,1,0,0,0,0] 

def test_unimodal(binary_unimodal):
    inflections = np.array([0,5,10,17,22])
    yhat, errors = 0

    