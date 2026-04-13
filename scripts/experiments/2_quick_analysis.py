from pathlib import Path

import pandas as pd

from src.result_analysis import ResultAnalysis

OUTPUT_DIRECTORY = Path('./assets/results')

ds = 'monash_m3_monthly'
# data_name, group, h = 'Gluonts', 'electricity_weekly', 12

results = pd.read_csv(f'{OUTPUT_DIRECTORY}/{ds},ensemble-fcst.csv', parse_dates=['ds'])

evaluator = ResultAnalysis('smape')

results = evaluator.map_forecasting_horizon_col(results)
# results = results.query('horizon<3')

sc_all = evaluator.overall_score(results)
sc_uid = evaluator.score_by_group(results, 'unique_id')
sc_es = evaluator.exp_shortfall(sc_uid, 0.9)
sc_horizon = evaluator.score_by_group(results, 'horizon')

print(sc_all.sort_values())
print(sc_uid.median().sort_values())
print(sc_uid.mean().sort_values())
print(sc_es.sort_values())
print(sc_horizon.iloc[0, :].sort_values())
print(sc_horizon.iloc[-1, :].sort_values())
print(sc_horizon.mean().sort_values())
print(sc_uid.rank(axis=1).mean().sort_values())

sc_horizon.head(12).mean().sort_values()
sc_horizon.tail(12).mean().sort_values()
