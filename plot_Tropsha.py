from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import plotly.graph_objects as go

res_path= '/home/kmurtazalieva/Downloads/scams/Results'
METRICS = ['AUC_LB', 'AUC', 'AUC_UB', 'Accuracy', 'F1_test', 'MCC_test']

to_plot = {'RF_N_SP1_SS': 'RF, No sampling, SCAMS_balanced_with_positive.csv, Scaffold splitter',
           'RF_ADASYN_SF_SS': 'RF, ADASYN, SCAMS_filtered.csv, Scaffold splitter',
           'RF_SMOTE_SF_SS': 'RF, SMOTE, SCAMS_filtered.csv, Scaffold splitter',
           'RF_N_SP1_TT1': 'RF, No sampling, SCAMS_filtered.csv, Scaffold splitter',
           'RF_CNN_SF_SS': 'RF, CondensedNearestNeighbour, SCAMS_filtered.csv, Scaffold splitter',
           'RF_SMOTE_SP2_SS': 'RF, SMOTE, SCAMS_added_positives_653_1043.csv, Scaffold splitter',
           'RF_ADASYN_SP2_SS': 'RF, ADASYN, SCAMS_added_positives_653_1043.csv, Scaffold splitter'}
# RF_SMOTE_SP2_SS': 'RandomForestClassifier, SMOTE, SCAMS_added_positives_653_1043.csv, Scaffold splitter'

def make_plot(paret_pat, dir, title):
    path= Path(res_path)/ dir/ 't-test_stats.csv'
    t_test = pd.read_csv(path)
    t_test = t_test.set_index('Metrics')
    print(t_test)
    AL = t_test['Mean AL'].loc[METRICS]
    non_AL = t_test['Mean non AL'].loc[METRICS]
    # Make adaptive max score
    max_perf = max(max(AL), max(non_AL))
    if max_perf + max_perf*0.1 > 1:
        max_perf = 1
    else:
        max_perf = max_perf + max_perf * 0.1

    radar = go.Figure()
    radar.add_trace(go.Scatterpolar(
        r=AL,
        theta=METRICS,
        fill='toself',
        name='AL strategy'
    ))
    radar.add_trace(go.Scatterpolar(
        r=non_AL,
        theta=METRICS,
        fill='toself',
        name='Non-AL strategy'
    ))
    radar.update_layout(
        title=title,
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max_perf]
            )),
        showlegend=True
    )
    radar_plot_path = Path(res_path)/ '{}.png'.format(dir)
    radar.write_image(str(radar_plot_path))

for dir_name, path in to_plot.items():
    make_plot(res_path, dir_name, path)



