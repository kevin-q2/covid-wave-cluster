import pandas as pd
import numpy as np

county_centers = pd.read_csv("../data/county_centers.txt", sep = ',')
data = pd.read_csv("data/county/infections.csv", index_col = 0)
county_names = data.columns

translator = {}
for cname in county_names:
    try:
        state_code = int(cname[6:8])
        county_code = int(cname[8:])
        state = county_centers.loc[county_centers.STATEFP == state_code]
        county = state.loc[state.COUNTYFP == county_code]
        full_name = county.COUNAME.iloc[0]
        translator[full_name] = cname
    except:
        pass
    

county_centers.COUNAME = [translator.get(i) for i in county_centers.COUNAME]
county_centers.drop(['STATEFP', 'COUNTYFP', 'STNAME', 'POPULATION'], axis = 1, inplace = True)
county_centers.to_csv("data/county/centers.csv")