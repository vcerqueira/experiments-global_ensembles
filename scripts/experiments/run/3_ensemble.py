import warnings
from pathlib import Path

import pandas as pd

from metaforecast.ensembles.mlpol import MLpol
from metaforecast.ensembles.mlewa import MLewa
from metaforecast.ensembles.ade import ADE
from metaforecast.ensembles.static import LossOnTrain, BestOnTrain, EqAverage
from metaforecast.ensembles.windowing import Windowing

from src.loaders import ChronosDataset, LongHorizonDatasetR
from src.config import TRIM_R

warnings.filterwarnings("ignore")

# ---- data loading and partitioning
target = 'monash_tourism_quarterly'
df, horizon, n_lags, freq, seas_len = ChronosDataset.load_everything(target)
# df, horizon, _, freq, seas_len = LongHorizonDatasetR.load_everything(target, resample_to='D')

# df['unique_id'].value_counts().value_counts().sort_index()
# from pprint import pprint
# dt = ChronosDataset.get_chronos_datasets_names()
# pprint(dt)

RESULTS_PATH = Path('../../../assets/results_cv')
# RESULTS_PATH = Path('./assets/results_cv')
ENSEMBLE_RESULTS_PATH = Path('../../../assets/results')
# ENSEMBLE_RESULTS_PATH = Path('./assets/results')

train, test = ChronosDataset.time_wise_split(df, horizon)

if __name__ == '__main__':

    # ---- model setup
    fcst_cv = pd.read_csv(RESULTS_PATH / f'{target},insample-base-fcst.csv', parse_dates=['ds'])
    fcst = pd.read_csv(RESULTS_PATH / f'{target},base-fcst.csv', parse_dates=['ds'])

    fcst_cv.drop(columns=['index'], inplace=True)
    fcst.drop(columns=['index'], inplace=True)

    # ---- fitting ensembles
    combiners_by_uid = {
        'ADE': ADE(freq=freq, meta_lags=list(range(1, n_lags + 1)), trim_ratio=TRIM_R, trim_by_uid=True),
        'MLpol': MLpol(loss_type='square', gradient=True, trim_ratio=TRIM_R, weight_by_uid=True),
        'MLewa': MLewa(loss_type='square', gradient=True, trim_ratio=TRIM_R, weight_by_uid=True),
        'LossOnTrain': LossOnTrain(trim_ratio=TRIM_R, weight_by_uid=True),
        'BestOnTrain': BestOnTrain(select_by_uid=True),
        'EqAverage': EqAverage(select_by_uid=True, trim_ratio=TRIM_R),
        'Windowing': Windowing(freq=freq, trim_ratio=TRIM_R, select_best=False, weight_by_uid=True),
        'BLAST': Windowing(freq=freq, trim_ratio=TRIM_R, select_best=True, weight_by_uid=True),
    }

    combiners_uncond = {
        'ADE': ADE(freq=freq, meta_lags=list(range(1, n_lags + 1)), trim_ratio=TRIM_R, trim_by_uid=False),
        'MLpol': MLpol(loss_type='square', gradient=True, trim_ratio=TRIM_R, weight_by_uid=False),
        'MLewa': MLewa(loss_type='square', gradient=True, trim_ratio=TRIM_R, weight_by_uid=False),
        'LossOnTrain': LossOnTrain(trim_ratio=TRIM_R, weight_by_uid=False),
        'BestOnTrain': BestOnTrain(select_by_uid=False),
        'EqAverage': EqAverage(select_by_uid=False, trim_ratio=TRIM_R),
        'Windowing': Windowing(freq=freq, trim_ratio=TRIM_R, select_best=False, weight_by_uid=False),
        'BLAST': Windowing(freq=freq, trim_ratio=TRIM_R, select_best=True, weight_by_uid=False),
    }

    for k in combiners_by_uid:
        print(k, "Unconditional")
        combiners_uncond[k].fit(fcst_cv)
        print(k, "by UID")
        combiners_by_uid[k].fit(fcst_cv)

    # ---- test forecasts
    print('...Combine forecasts')
    ensembles = {}
    for k in combiners_by_uid:
        print(k)
        if k == 'ADE':
            fc_uid = combiners_by_uid[k].predict(fcst, train=train, h=horizon)
            fc = combiners_uncond[k].predict(fcst, train=train, h=horizon)
        else:
            fc_uid = combiners_by_uid[k].predict(fcst)
            fc = combiners_uncond[k].predict(fcst)

        ensembles[k] = fc
        ensembles[f'{k}(uid)'] = fc_uid

    ensembles_df = pd.DataFrame(ensembles)

    fcst_df = pd.concat([fcst, ensembles_df], axis=1)
    fcst_df = fcst_df.merge(test, on=['unique_id', 'ds'])

    fcst.to_csv(ENSEMBLE_RESULTS_PATH / f'{target},ensemble-fcst.csv', index=False)
