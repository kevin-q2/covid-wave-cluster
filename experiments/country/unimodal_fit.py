import sys
import os
import numpy as np
from wave_cluster import *
sys.path.append("experiments/country/")
from data_load import load_data

cpu_count = 16

infections = load_data()
data_array = infections.to_numpy()

wave_mod = Unimodal(penalty_by_length = 15)
dist_mod = DynamicTimeWarp(
    mult_penalty = np.array([1.0,1.0,1.0], dtype=np.float64),
    add_penalty = np.array([1/7,1/7,0.0], dtype=np.float64),
    normalize = True
)
wave_pool = WavePool(
    wave_module = wave_mod,
    distance_module = dist_mod,
    fit_waves_ = True,
    fit_distances_ = True,
    mask = True,
    threshold = 90,
    cpu_count = cpu_count
)

wave_pool.fit(data_array)
wave_pool.save_pool('experiments/country/data/unimodal_pool.npz')
wave_pool.save_distances('experiments/country/data/unimodal_distances.npz')