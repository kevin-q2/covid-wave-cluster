import sys
import os
import numpy as np
from wave_cluster import *
sys.path.append("experiments/state/")
from data_load import load_data

cpu_count = 16

data = load_data()
data_array = data.to_numpy()

wave_mod = Unimodal(penalty = 0.01, normalize = True)
dist_mod = DynamicTimeWarp(
    mult_penalty = [1.0,1.0,1.0],
    add_penalty = [0.05,0.05,0.0],
    normalize = True
)
wave_pool = WavePool(
    wave_module = wave_mod,
    distance_module = dist_mod,
    fit_waves_ = True,
    fit_distances_ = True,
    mask = True,
    cpu_count = cpu_count
)

wave_pool.fit(data_array)
wave_pool.save_pool('experiments/state/data/unimodal_pool.npz')
wave_pool.save_distances('experiments/state/data/unimodal_distances.npz')