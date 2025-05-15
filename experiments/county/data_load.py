import sys
import numpy as np
import pandas as pd
from wave_cluster import *


# This script's purpose is to load the Covid-19 dataset!
# Only input parameter is to decide whether or not to drop a larger subset of the locations

def load_data():
    # US state level infections, population, geography, and government response data
    infections = pd.read_csv("data/county/infections.csv", index_col = 0)
    population = pd.read_csv("data/county/population.csv", index_col = 0)

    # Normalization of each locations time-series to be cases per 100,000 persons
    norm = infections.apply(lambda x: x/(population.loc[x.name].iloc[0]))
    infections = norm * 100000 # cases per 100,000

    # Windowed average smoothing of each time-series 
    # Using average of 7 days in front and behind burrent time-stamp, 
    # and repeating 3 times for a smoother result. 
    front = 7
    back = 7
    smoothed = window_average_2d(infections.to_numpy(), front = front, back = back)
    smoothed = window_average_2d(smoothed, front = front, back = back)
    smoothed = window_average_2d(smoothed, front = front, back = back)

    # remove any negative entries (reporting errors still present after smoothing)
    smoothed[smoothed < 0] = 0

    # fix the indexes
    smoothed_index = pd.to_datetime(infections.index[front*3:-back*3])
    infections = pd.DataFrame(smoothed, index = smoothed_index, columns = infections.columns)
    return infections