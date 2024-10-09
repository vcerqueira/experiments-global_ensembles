import os

import pandas as pd

from utils.workflows.results import ResultAnalysis

METRIC = 'smape'
DIR = 'assets/results'

# ----- reading results

scores, scores_by_uid, scores_1h, scores_fh = [], [], [], []
for file in os.listdir(DIR):
    if file == '.DS_Store':
        continue

    print(file)

    fp = f'{DIR}/{file}'

    dt = file.split('.')[0].split('_')

    results = pd.read_csv(fp, parse_dates=['ds'])

    idx = results.columns.str.startswith('LossOnTra')

    results = results.iloc[:, ~idx]

    evaluator = ResultAnalysis(METRIC)

    sc_all = evaluator.overall_score(results)
    sc_uid = evaluator.score_by_group(results, 'unique_id')

    results = evaluator.map_forecasting_horizon_col(results)

    sc_h = evaluator.score_by_group(results, 'horizon')

    sc_firsth = sc_h.head(1).mean()
    sc_fullh = sc_h.mean()

    sc_all['Data'] = dt[0]
    sc_all['Frequency'] = dt[-2]
    sc_uid['Data'] = dt[0]
    sc_uid['Frequency'] = dt[-2]
    sc_firsth['Data'] = dt[0]
    sc_firsth['Frequency'] = dt[-2]
    sc_fullh['Data'] = dt[0]
    sc_fullh['Frequency'] = dt[-2]

    scores.append(sc_all)
    scores_by_uid.append(sc_uid)
    scores_1h.append(sc_firsth)
    scores_fh.append(sc_fullh)

scores_df = pd.concat(scores, axis=1).T
scores_uid_df = pd.concat(scores_by_uid, axis=0).reset_index()
scores_uid_df = scores_uid_df.rename(columns={'index': 'unique_id'})
scores_1h_df = pd.concat(scores_1h, axis=1).T
scores_fh_df = pd.concat(scores_fh, axis=1).T

MAPPER = {
    'MLP1': 'MLP(3L)',
    'DilatedRNN': 'DLSTM',
    'DilatedRNN1': 'DGRU',
    'SeasonalNaive': 'SNaive',
}
scores_df = scores_df.rename(columns=MAPPER)
scores_uid_df = scores_uid_df.rename(columns=MAPPER)
scores_1h_df = scores_1h_df.rename(columns=MAPPER)
scores_fh_df = scores_fh_df.rename(columns=MAPPER)

scores_df.to_csv('assets/scores_avg.csv', index=False)
scores_uid_df.to_csv('assets/scores_uid.csv', index=False)
scores_1h_df.to_csv('assets/scores_1h.csv', index=False)
scores_fh_df.to_csv('assets/scores_fh.csv', index=False)
