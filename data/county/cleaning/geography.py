import pandas as pd
import numpy as np

# Read the geography dataset, taken straight from the source:
# https://health.google.com/covid-19/open-data/raw-data
geo = pd.read_csv("../data/geography.csv", index_col = 0)

# Find coordinates for each county in the infections dataset.
data = pd.read_csv("data/county/infections.csv", index_col = 0)
locations = data.columns
county_geo = geo.loc[locations,["latitude", "longitude"]]
county_geo.to_csv("data/county/geography.csv")