import sys
import numpy as np
import pandas as pd
from wave_cluster import *


# This script's purpose is to load the Covid-19 dataset!
# Only input parameter is to decide whether or not to drop a larger subset of the locations

def load_data():
    # US state level demographics and Covid-19 Time-Series data (# new cases every day)
    demographics = pd.read_csv("data/demographics.csv", index_col = 0)
    data = pd.read_csv("data/state_daily.csv", index_col = 0)

    # dropping the first 39 days for unreliable data 
    data = data.iloc[39:,:]

    # Dropping non-state and non-continental US locations.
    dr = ['US_AK', 'US_HI', 'US_DC', 'US_PR', 'US_VI', 'US_MP', 'US_GU', 'US_AS']
    data = data.drop(dr, axis = 1)

    # Normalization of each locations time-series to be cases per 100,000 persons
    population = demographics.loc[data.columns,"population"]
    norm_data = data.apply(lambda x: x/population[x.name])
    data = norm_data * 100000 # cases per 100,000

    # Windowed average smoothing of each time-series 
    # Using average of 7 days in front and behind burrent time-stamp, 
    # and repeating 3 times for a smoother result. 
    front = 7
    back = 7
    smooth_data = window_average_2d(data.to_numpy(), front = front, back = back)
    smooth_data = window_average_2d(smooth_data, front = front, back = back)
    smooth_data = window_average_2d(smooth_data, front = front, back = back)

    # fix the index
    smoothed_index = pd.to_datetime(data.index[front*3:-back*3])
    data = pd.DataFrame(smooth_data, index = smoothed_index, columns = data.columns)

    # remove any negative entries (reporting errors still present after smoothing)
    data[data < 0] = 0

    return data