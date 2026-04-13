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

RESULTS_PATH = Path('../../../assets/results_cv')
# RESULTS_PATH = Path('./assets/results_cv')
ENSEMBLE_RESULTS_PATH = Path('../../../assets/results')
# ENSEMBLE_RESULTS_PATH = Path('./assets/results')

train, test = ChronosDataset.time_wise_split(df, horizon)


def override_ds_and_merge(fcst, trues):
    fcst = fcst.sort_values(['unique_id', 'ds'])
    trues = trues.sort_values(['unique_id', 'ds'])

    fcst['temp_idx'] = fcst.groupby('unique_id').cumcount()
    trues['temp_idx'] = trues.groupby('unique_id').cumcount()

    fcst = fcst.drop(columns=['ds'])

    result = pd.merge(fcst, trues, on=['unique_id', 'temp_idx'])

    result = result.drop(columns=['temp_idx'])

    return result


CATBOOST_PARAMS = {
    'eval_metric': 'MultiRMSE',
    'loss_function': 'MultiRMSE',
    'od_type': 'Iter',
    'allow_writing_files': False,
    'task_type': 'CPU',
    'verbose': False}

if __name__ == '__main__':

    # ---- model setup
    fcst_cv = pd.read_csv(RESULTS_PATH / f'{target},insample-base-fcst.csv', parse_dates=['ds'])
    fcst = pd.read_csv(RESULTS_PATH / f'{target},base-fcst.csv', parse_dates=['ds'])

    fcst_cv.drop(columns=['index'], inplace=True)
    fcst.drop(columns=['index'], inplace=True)

    # todo need to have this warning in metaforecast
    min_sample_size = fcst_cv['unique_id'].value_counts().min()

    if min_sample_size < int(n_lags/2):
        n_lags_ = min_sample_size
    else:
        n_lags_ = n_lags

    # todo add QE in metaforecast
    freq_ = 'Q' if freq == 'QE' else freq

    # ---- fitting ensembles
    combiners_by_uid = {
        'ADE': ADE(freq=freq_,
                   meta_lags=list(range(1, n_lags_)),
                   meta_model=CatBoostRegressor(**CATBOOST_PARAMS),
                   trim_ratio=TRIM_R, trim_by_uid=True),
        'MLpol': MLpol(loss_type='square', gradient=True, trim_ratio=TRIM_R, weight_by_uid=True),
        'MLewa': MLewa(loss_type='square', gradient=True, trim_ratio=TRIM_R, weight_by_uid=True),
        'LossOnTrain': LossOnTrain(trim_ratio=TRIM_R, weight_by_uid=True),
        'BestOnTrain': BestOnTrain(select_by_uid=True),
        'EqAverage': EqAverage(select_by_uid=True, trim_ratio=TRIM_R),
        'Windowing': Windowing(freq=freq_, trim_ratio=TRIM_R, select_best=False, weight_by_uid=True),
        'BLAST': Windowing(freq=freq_, trim_ratio=TRIM_R, select_best=True, weight_by_uid=True),
    }

    combiners_uncond = {
        'ADE': ADE(freq=freq_, meta_lags=list(range(1, n_lags_)),
                   trim_ratio=TRIM_R,
                   meta_model=CatBoostRegressor(**CATBOOST_PARAMS),
                   trim_by_uid=False),
        'MLpol': MLpol(loss_type='square', gradient=True, trim_ratio=TRIM_R, weight_by_uid=False),
        'MLewa': MLewa(loss_type='square', gradient=True, trim_ratio=TRIM_R, weight_by_uid=False),
        'LossOnTrain': LossOnTrain(trim_ratio=TRIM_R, weight_by_uid=False),
        'BestOnTrain': BestOnTrain(select_by_uid=False),
        'EqAverage': EqAverage(select_by_uid=False, trim_ratio=TRIM_R),
        'Windowing': Windowing(freq=freq_, trim_ratio=TRIM_R, select_best=False, weight_by_uid=False),
        'BLAST': Windowing(freq=freq_, trim_ratio=TRIM_R, select_best=True, weight_by_uid=False),
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
    print(fcst_df)
    print(test)
    # fcst = fcst_df.merge(test, on=['unique_id', 'ds'])
    fcst = override_ds_and_merge(fcst_df, test)
    print(fcst)

    fcst.to_csv(ENSEMBLE_RESULTS_PATH / f'{target},ensemble-fcst.csv', index=False)
