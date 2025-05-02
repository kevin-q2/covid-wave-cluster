import pandas as pd
import numpy as np

state_centers = pd.read_csv("../data/state_centers.txt", sep = ',', index_col = 0)

translator = {
    'US_AK': 'Alaska',
    'US_AL':'Alabama',
    'US_AR':'Arkansas',
    'US_AZ':'Arizona',
    'US_CA':'California',
    'US_CO':'Colorado',
    'US_CT':'Connecticut',
    'US_DC':'District of Columbia',
    'US_DE':'Delaware', 
    'US_FL':'Florida',
    'US_GA':'Georgia',
    'US_HI':'Hawaii',
    'US_IA': 'Iowa',
    'US_ID':'Idaho',
    'US_IL':'Illinois',
    'US_IN':'Indiana',
    'US_KS':'Kansas',
    'US_KY':'Kentucky',
    'US_LA':'Louisiana',
    'US_MA':'Massachusetts',
    'US_MD':'Maryland',
    'US_ME':'Maine',
    'US_MI':'Michigan',
    'US_MN':'Minnesota',
    'US_MO':'Missouri',
    'US_MS':'Mississippi',
    'US_MT':'Montana',
    'US_NC':'North Carolina',
    'US_ND':'North Dakota',
    'US_NE':'Nebraska',
    'US_NH':'New Hampshire',
    'US_NJ':'New Jersey',
    'US_NM':'New Mexico',
    'US_NV':'Nevada',
    'US_NY':'New York',
    'US_OH':'Ohio',
    'US_OK':'Oklahoma',
    'US_OR':'Oregon',
    'US_PA':'Pennsylvania', 
    'US_PR':'Puerto Rico',
    'US_RI':'Rhode Island',
    'US_SC':'South Carolina',
    'US_SD':'South Dakota',
    'US_TN':'Tennessee',
    'US_TX':'Texas',
    'US_UT':'Utah',
    'US_VA':'Virginia',
    'US_VT':'Vermont',
    'US_WA':'Washington',
    'US_WI':'Wisconsin',
    'US_WV':'West Virginia',
    'US_WY':'Wyoming'
}

translator_reverse = {i[1]:i[0] for i in translator.items()}
state_centers.STNAME = [translator_reverse[i] for i in state_centers.STNAME]
state_centers.drop('POPULATION', axis = 1, inplace = True)
state_centers.to_csv("data/state/centers.csv")