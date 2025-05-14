import pandas as pd
import numpy as np

# Read the geography dataset, taken straight from the source:
# https://health.google.com/covid-19/open-data/raw-data
geo = pd.read_csv("../data/geography.csv", index_col = 0)

# Find coordinates for each state in the infections dataset.
data = pd.read_csv("data/state/infections.csv", index_col = 0)
locations = data.columns
state_geo = geo.loc[locations,["latitude", "longitude"]]
state_geo.to_csv("data/state/geography.csv")