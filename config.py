import pandas as pd

feature_list_diff = ['Raw_Activity_Data', 'Activity_Change', 'Activity_Change_by_2_Hours', \
    'Rumination_Raw_Data', 'Weighted_Rumination_Change', 'Total_Rumination_Minutes_In_Last', \
    'Daily_Rumination', 'Weekly_Rumination_Average', 'new_health_index', 'Daily_activity']

feature_list_ease = ['Raw_Activity_Data', 'Activity_Change', 'Activity_Change_by_2_Hours', \
    'Rumination_Raw_Data', 'Weighted_Rumination_Change', 'Total_Rumination_Minutes_In_Last', \
    'Daily_Rumination', 'Weekly_Rumination_Average', 'new_health_index', 'Daily_activity', \
    'LACT', 'parity']

feature_list_same = ['LACT', 'PDIM', 'PDOPN', 'PREFR', 'PTOTF', 'PTOTM', 'PTOTS', 'parity']

feature_list_once = ['LACT', 'PDIM', 'PDOPN', 'PREFR', 'PTOTF', 'PTOTM', 'PTOTS', \
    'Raw_Activity_Data', 'Activity_Change', 'Activity_Change_by_2_Hours', \
    'Rumination_Raw_Data', 'Weighted_Rumination_Change', 'Total_Rumination_Minutes_In_Last', \
    'Daily_Rumination', 'Weekly_Rumination_Average', 'new_health_index', 'Daily_activity', \
    'parity']

feature_list_all = feature_list_same[:]
for feat in feature_list_diff:
    for num in ['0','1','2']:
        feature_list_all.append(feat+num)

mean_feat = pd.read_pickle('mean2.pkl')
