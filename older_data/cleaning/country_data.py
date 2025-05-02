import pandas as pd 
import numpy as np 

# read entire dataset
epi = pd.read_csv('data/raw/epidemiology.csv', index_col = 0)

# Filter to country level data,
# location_key is a 2-letter country code
pattern = r'[A-Z]{2}'
country_epi = epi.loc[epi['location_key'].str.fullmatch(pattern, na = False)]

# Form a new dataframe
dates = np.sort(country_epi.index.value_counts().index)
columns = np.sort(country_epi.location_key.value_counts().index)
date_index = {dates[i]:i for i in range(len(dates))}

country_data = np.zeros((len(dates), len(columns)))
country_data[:] = np.nan
for l in range(len(columns)):
    confirmed = country_epi.loc[country_epi.location_key == columns[l]].new_confirmed
    idx = [date_index[i] for i in confirmed.index]
    country_data[idx, l] = confirmed
        
        
country_data = pd.DataFrame(country_data, index = dates, columns = columns)
country_data.to_csv('data/country_daily.csv')
        
