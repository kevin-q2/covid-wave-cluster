import pandas as pd 
import numpy as np

# Read the demographic dataset, taken straight from the source:
# https://health.google.com/covid-19/open-data/raw-data
demographics = pd.read_csv("../data/demographics.csv", index_col = 0)

# Find population for each state in the infections dataset.
data = pd.read_csv("data/county/infections.csv", index_col = 0)
locations = data.columns
population = demographics.loc[locations,"population"]
population.to_csv("data/county/population.csv")