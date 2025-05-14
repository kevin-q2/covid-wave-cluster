import pandas as pd 
import numpy as np 

#oxn = pd.read_csv("data/raw/OxCGRT_compact_national_v1.csv", index_col = 0)
oxn = pd.read_csv('../covid-policy-dataset/data/OxCGRT_compact_national_v1.csv', index_col = 0)
oxn.Date = pd.to_datetime(oxn.Date, format = "%Y%m%d")
dates_used = oxn.Date.value_counts().index.sort_values()

iso2_to_iso3 = {
    'AD': 'AND', 'AL': 'ALB', 'AM': 'ARM', 'AT': 'AUT', 'AZ': 'AZE',
    'BA': 'BIH', 'BE': 'BEL', 'BG': 'BGR', 'BY': 'BLR', 'HR': 'HRV',
    'CY': 'CYP', 'CZ': 'CZE', 'DK': 'DNK', 'EE': 'EST', 'FI': 'FIN',
    'FR': 'FRA', 'DE': 'DEU', 'GR': 'GRC', 'HU': 'HUN', 'IE': 'IRL',
    'IT': 'ITA', 'LV': 'LVA', 'LT': 'LTU', 'LU': 'LUX', 'MT': 'MLT',
    'NL': 'NLD', 'PL': 'POL', 'PT': 'PRT', 'RO': 'ROU', 'SK': 'SVK',
    'SI': 'SVN', 'ES': 'ESP', 'SE': 'SWE', 'GB': 'GBR', 'GE': 'GEO',
    'IS': 'ISL', 'KZ': 'KAZ', 'LI': 'LIE', 'MD': 'MDA', 'MC': 'MCO',
    'ME': 'MNE', 'MK': 'MKD', 'NO': 'NOR', 'RU': 'RUS', 'SM': 'SMR',
    'RS': 'SRB', 'CH': 'CHE', 'TR': 'TUR', 'UA': 'UKR'
}

'''
EU = ['AD', 'AL', 'AM','AT','AZ','BA', 'BE', 'BG','BY', 'HR', 'CY', 'CZ', 'DK', 'EE', 'FI', 'FR',
     'DE', 'GR', 'HU', 'IE', 'IT', 'LV', 'LT', 'LU', 'MT', 'NL', 
     'PL', 'PT', 'RO', 'SK', 'SI', 'ES', 'SE', 'GB', 'GE', 'IS', 'KZ', 'LI', 'MD', 'MC', 'ME',
     'MK', 'NO', 'RU', 'SM', 'RS', 'CH', 'TR', 'UA']
'''

EU = ['AD', 'AL', 'AT','AZ','BA', 'BE', 'BG','BY', 'HR', 'CY', 'CZ', 'DK', 'EE', 'FI', 'FR',
     'DE', 'GR', 'HU', 'IE', 'IT', 'LV', 'LT', 'LU', 'MT', 'NL', 
     'PL', 'PT', 'RO', 'SK', 'SI', 'ES', 'SE', 'GB', 'GE', 'IS', 'KZ', 'LI', 'MD', 'MC',
     'NO', 'RU', 'SM', 'RS', 'CH', 'TR', 'UA']

containment_health = [oxn.loc[oxn.CountryCode == iso2_to_iso3[EU[i]]].ContainmentHealthIndex_Average.to_numpy() 
                   for i in range(len(EU))]

containment_health = np.array(containment_health).T
containment_health = pd.DataFrame(containment_health, index = dates_used, columns = EU)
containment_health.to_csv('data/country/gov_response.csv')