import pandas as pd 
import numpy as np 

# read entire dataset
epi = pd.read_csv('data/raw/epidemiology.csv', index_col = 0)

# filter to US counties,
# location_key is a 2-letter state code followed by a 5-digit county code
pattern = r'US_[A-Z]{2}_[0-9]{5}'
county_epi = epi.loc[epi['location_key'].str.match(pattern, na = False)]

# form a new dataframe
dates = np.sort(county_epi.index.value_counts().index)
columns = np.sort(county_epi.location_key.value_counts().index)
date_index = {dates[i]:i for i in range(len(dates))}

county_data = np.zeros((len(dates), len(columns)))
county_data[:] = np.nan
for l in range(len(columns)):
    confirmed = county_epi.loc[county_epi.location_key == columns[l]].new_confirmed
    idx = [date_index[i] for i in confirmed.index]
    county_data[idx, l] = confirmed
        
        
county_data = pd.DataFrame(county_data, index = dates, columns = columns)
county_data.to_csv('data/county_daily.csv')
        
