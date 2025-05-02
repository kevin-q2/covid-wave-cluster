vacc = pd.read_csv('../../data/us_state_vaccinations.csv', index_col = 0)
vacc.date = pd.to_datetime(vacc.date)


idx = pd.date_range('01-01-2020', '09-16-2022')
vac_locs = vacc.index.value_counts().index
vacc.isnull().values.any()
vacc = vacc.fillna(0)


new_idx = []
for l in vac_locs:
    for i in idx:
        new_idx.append(l)
        
new_date = []
for l in vac_locs:
    for i in idx:
        new_date.append(i)
        
empty_frame = [[] for i in range(len(new_idx))]
for l in range(len(vac_locs)):
    for i in range(len(idx)):
        date_id = idx[i]
        try:
            empty_frame[l*len(idx) + i] += vacc.loc[vacc.date == date_id].loc[vac_locs[l]].to_list()
        except KeyError:
            empty_frame[l*len(idx) + i] += [date_id] + [0]*(len(vacc.columns) - 1)
            
empty_frame = pd.DataFrame(empty_frame)
empty_frame.columns = vacc.columns
empty_frame.index = new_idx
#empty_frame.date = new_date


col_idx = {'cumulative_vaccinations': 2, 'cumulative_full_vaccinations': 4, 'cumulative_doses_vaccinations':6}
for col in ['cumulative_vaccinations', 'cumulative_full_vaccinations', 'cumulative_doses_vaccinations']:
    for l in range(len(vac_locs)):
        cloc = vac_locs[l]
        #loc_idx = vacc.loc[cloc].index
        prev_value = empty_frame.loc[cloc, col].iloc[0]
        for i in range(1, len(idx)):
            current_value = empty_frame.loc[cloc, col].iloc[i]
            if current_value < prev_value:
                empty_frame.iloc[l*len(idx) + i, col_idx[col]] = prev_value
            else:
                prev_value = current_value
                
                
empty_frame.to_csv('../../data/us_state_vaccinations.csv')