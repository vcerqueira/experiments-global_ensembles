import warnings
from pathlib import Path

import pandas as pd

from catboost import CatBoostRegressor
from metaforecast.ensembles.mlpol import MLpol
from metaforecast.ensembles.mlewa import MLewa
from metaforecast.ensembles.ade import ADE
from metaforecast.ensembles.static import LossOnTrain, BestOnTrain, EqAverage
from metaforecast.ensembles.windowing import Windowing

from src.loaders import ChronosDataset, LongHorizonDatasetR
from src.config import TRIM_R

warnings.filterwarnings("ignore")

# ---- data loading and partitioning
target = 'monash_m3_monthly'
df, horizon, n_lags, freq, seas_len = ChronosDataset.load_everything(target)
# df, horizon, _, freq, seas_len = LongHorizonDatasetR.load_everything(target, resample_to='D')

# df['unique_id'].value_counts().value_counts().sort_index()
# from pprint import pprint
# dt = ChronosDataset.get_chronos_datasets_names()
# pprint(dt)

# RESULTS_PATH = Path('../../../assets/results_cv')
RESULTS_PATH = Path('./assets/results_cv')
# ENSEMBLE_RESULTS_PATH = Path('../../../assets/results')
ENSEMBLE_RESULTS_PATH = Path('./assets/results')

train, test = ChronosDataset.time_wise_split(df, horizon)

CATBOOST_PARAMS = {
    'eval_metric': 'MultiRMSE',
    'loss_function': 'MultiRMSE',
    'od_type': 'Iter',
    'task_type': 'CPU',
    'verbose': False}

model = CatBoostRegressor(**CATBOOST_PARAMS)



fcst_cv = pd.read_csv(RESULTS_PATH / f'{target},insample-base-fcst.csv', parse_dates=['ds'])
fcst = pd.read_csv(RESULTS_PATH / f'{target},base-fcst.csv', parse_dates=['ds'])

fcst_cv.drop(columns=['index'], inplace=True)
fcst.drop(columns=['index'], inplace=True)

# m = ADE(freq=freq, meta_lags=list(range(1, n_lags + 1)), trim_ratio=TRIM_R, trim_by_uid=True)
m = ADE(freq=freq, meta_lags=list(range(1, 3)),
        trim_ratio=TRIM_R,
        trim_by_uid=False,
        meta_model=model)

m.fit(fcst_cv)
m.predict(fcst, train=train, h=horizon)




