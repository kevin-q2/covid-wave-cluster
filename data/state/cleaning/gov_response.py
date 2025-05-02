import pandas as pd
import numpy as np

# Read entire dataset, taken straight from the source:
# https://health.google.com/covid-19/open-data/raw-data
epi = pd.read_csv('../data/oxford-government-response.csv', index_col = 0)

# Filter to US States and Territories,
# for which location_key is a 2-letter state code
pattern = r'^US_[A-Z]{2}$'
filtered_epi = epi.loc[epi['location_key'].str.match(pattern, na = False)]

# Find the unique dates (row indices) and locations (column indices)
dates = np.sort(filtered_epi.index.value_counts().index)
columns = np.sort(filtered_epi.location_key.value_counts().index)
date_index = {dates[i]:i for i in range(len(dates))}

# Fill a data array with daily, new confirmed cases
data = np.zeros((len(dates), len(columns)))
data[:] = np.nan
for l, col in enumerate(columns):
    confirmed = filtered_epi.loc[filtered_epi.location_key == col].new_confirmed
    idx = [date_index[i] for i in confirmed.index]
    data[idx, l] = confirmed

# Save as a dataframe        
data = pd.DataFrame(data, index = dates, columns = columns)
data.to_csv('data/state/gov_response.csv')