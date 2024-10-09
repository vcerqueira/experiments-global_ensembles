import pandas as pd

from utils.workflows.results import ResultAnalysis
from utils.config import OUTPUT_DIRECTORY

data_name, group, h = 'M3', 'Yearly', 4
# data_name, group, h = 'Gluonts', 'electricity_weekly', 12

results = pd.read_csv(f'{OUTPUT_DIRECTORY}/{data_name}_{group}_{h}.csv', parse_dates=['ds'])

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
