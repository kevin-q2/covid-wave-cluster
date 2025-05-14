import pandas as pd
import numpy as np

# Read from the Oxford covid-policy-dataset
gov = pd.read_csv(
    '../covid-policy-dataset/data/OxCGRT_compact_subnational_v1.csv'
)
us = gov.loc[gov.CountryCode == 'USA']
locations = us.RegionCode.value_counts().index
# Using Alaska as an example
ak = us.loc[us.RegionCode == 'US_AK']
dates = pd.to_datetime(ak.Date, format = "%Y%m%d")
date_index = {dates.iloc[i]:i for i in range(len(dates))}

# Fill a data array with daily, new confirmed cases
data = np.zeros((len(dates), len(locations)))
data[:] = np.nan
for l, loc in enumerate(locations):
    loc_data = us.loc[us.RegionCode == loc]
    containment_health = loc_data.ContainmentHealthIndex_Average
    loc_dates = pd.to_datetime(loc_data.Date, format = "%Y%m%d")
    idx = [date_index[i] for i in loc_dates]
    data[idx, l] = containment_health

# Fix the index and save as a dataframe        
gov_response = pd.DataFrame(data, index = dates, columns = locations)
infections = pd.read_csv("data/state/infections.csv", index_col = 0)
gov_response = gov_response.loc[infections.index, infections.columns]
gov_response.to_csv('data/state/gov_response.csv')