import warnings
from pathlib import Path

import pandas as pd
from metaforecast.ensembles import (MLpol,
                                    MLewa,
                                    MLprod,
                                    FixedShare,
                                    ADE,
                                    LossOnTrain,
                                    BestOnTrain,
                                    EqAverage,
                                    Windowing,
                                    BOA,
                                    OGD,
                                    Ridge)

from src.loaders import ChronosDataset, LongHorizonDatasetR
from src.config import TRIM_R, DATASETS, LH_DATASETS, USE_TRAINING_LOSS

warnings.filterwarnings("ignore")

RESULTS_PATH = Path(__file__).resolve().parents[2] / "assets" / "results_cv"
ENSEMBLE_RESULTS_PATH = Path(__file__).resolve().parents[2] / "assets" / "results"

TRIM_CFG_T = {'trim_ratio': TRIM_R, 'weight_by_uid': True}
TRIM_CFG_F = {'trim_ratio': TRIM_R, 'weight_by_uid': False}


def override_ds_and_merge(fcst, trues):
    # Replace forecast ds with the test-set timestamps when they disagree.
    keys = ['unique_id', 'ds']
    fcst = fcst.sort_values(keys).drop(columns='ds')
    trues = trues.sort_values(keys)
    fcst = fcst.assign(h=fcst.groupby('unique_id').cumcount())
    trues = trues.assign(h=trues.groupby('unique_id').cumcount())
    return fcst.merge(trues, on=['unique_id', 'h']).drop(columns='h')


def _align_ds_keys(df):
    out = df[['unique_id', 'ds']].copy()
    out['unique_id'] = out['unique_id'].astype(str)
    out['ds'] = pd.to_datetime(out['ds']).astype('datetime64[ns]')
    return out.sort_values(['unique_id', 'ds']).reset_index(drop=True)


def assert_ds_match(fcst, trues):
    if not _align_ds_keys(fcst).equals(_align_ds_keys(trues)):
        raise ValueError('Forecast timestamps do not match the test set.')


def _harmonize_keys(df):
    df = df.copy()
    df['unique_id'] = df['unique_id'].astype(str)
    df['ds'] = pd.to_datetime(df['ds']).astype('datetime64[ns]')
    return df


def load_dataset(target):
    if target in LH_DATASETS:
        _, horizon, n_lags, _, _ = LongHorizonDatasetR.load_everything(target, resample_to='D')
        df, horizon, n_lags, freq, _ = LongHorizonDatasetR.load_everything(
            target,
            min_n_instances=2 * (n_lags + horizon),
            resample_to='D',
        )
    else:
        _, horizon, n_lags, _, _ = ChronosDataset.load_everything(target)
        df, horizon, _, freq, _ = ChronosDataset.load_everything(
            target,
            min_n_instances=2 * (n_lags + horizon),
        )

    train, test = ChronosDataset.time_wise_split(df, horizon)
    return train, test, horizon, n_lags, freq


def make_combiners(freq, n_lags, horizon):
    combiners_by_uid = {
        'ADE': ADE(freq=freq, meta_lags=n_lags, trim_by_uid=True, trim_ratio=TRIM_R, h=horizon),
        'MLpol': MLpol(loss_type='square', gradient=True, **TRIM_CFG_T),
        'MLewa': MLewa(loss_type='square', gradient=True, **TRIM_CFG_T),
        'MLprod': MLprod(loss_type='square', gradient=True, **TRIM_CFG_T),
        'BOA': BOA(loss_type='square', gradient=True, **TRIM_CFG_T),
        'OGD': OGD(loss_type='square', gradient=True, **TRIM_CFG_T),
        'FS': FixedShare(loss_type='square', gradient=True, **TRIM_CFG_T),
        'Ridge': Ridge(loss_type='square', gradient=True, **TRIM_CFG_T),
        'LossOnTrain': LossOnTrain(**TRIM_CFG_T),
        'BestOnTrain': BestOnTrain(select_by_uid=True),
        'EqAverage': EqAverage(select_by_uid=True, trim_ratio=TRIM_R),
        'Windowing': Windowing(freq=freq, select_best=False, **TRIM_CFG_T),
        'BLAST': Windowing(freq=freq, select_best=True, **TRIM_CFG_T),
    }

    combiners_uncond = {
        'ADE': ADE(freq=freq, meta_lags=n_lags, trim_by_uid=False, trim_ratio=TRIM_R, h=horizon),
        'MLpol': MLpol(loss_type='square', gradient=True, **TRIM_CFG_F),
        'MLewa': MLewa(loss_type='square', gradient=True, **TRIM_CFG_F),
        'MLprod': MLprod(loss_type='square', gradient=True, **TRIM_CFG_F),
        'BOA': BOA(loss_type='square', gradient=True, **TRIM_CFG_F),
        'OGD': OGD(loss_type='square', gradient=True, **TRIM_CFG_F),
        'FS': FixedShare(loss_type='square', gradient=True, **TRIM_CFG_F),
        'Ridge': Ridge(loss_type='square', gradient=True, **TRIM_CFG_F),
        'LossOnTrain': LossOnTrain(**TRIM_CFG_F),
        'BestOnTrain': BestOnTrain(select_by_uid=False),
        'EqAverage': EqAverage(select_by_uid=False, trim_ratio=TRIM_R),
        'Windowing': Windowing(freq=freq, select_best=False, **TRIM_CFG_F),
        'BLAST': Windowing(freq=freq, select_best=True, **TRIM_CFG_F),
    }

    return combiners_by_uid, combiners_uncond


if __name__ == '__main__':
    ENSEMBLE_RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    print(ENSEMBLE_RESULTS_PATH.absolute())

    for target in DATASETS:
        print(f"Running {target}")
        weight_tag = 'fitted' if USE_TRAINING_LOSS else 'cv'
        base_fp = RESULTS_PATH / f'{target},base-fcst.csv'
        insample_fp = RESULTS_PATH / f'{target},base-{weight_tag}-fcst.csv'
        out_fp = ENSEMBLE_RESULTS_PATH / f'{target},ensemble-fcst,weights-{weight_tag}.csv'

        if out_fp.exists():
            continue

        if not base_fp.exists() or not insample_fp.exists():
            print(f"Skipping {target}: missing CV forecasts")
            continue

        train, test, horizon, n_lags, freq = load_dataset(target)

        fcst_cv = pd.read_csv(insample_fp, parse_dates=['ds'])
        fcst = pd.read_csv(base_fp, parse_dates=['ds'])


        combiners_by_uid, combiners_uncond = make_combiners(freq, n_lags, horizon)

        for k in combiners_by_uid:
            print(k, "Unconditional")
            combiners_uncond[k].fit(fcst_cv)
            print(k, "by UID")
            combiners_by_uid[k].fit(fcst_cv)

        print('...Combine forecasts')
        ensembles = {}
        for k in combiners_by_uid:
            print(k)
            if k == 'ADE':
                fc_uid = combiners_by_uid[k].predict(fcst, train=train)
                fc = combiners_uncond[k].predict(fcst, train=train)
            else:
                fc_uid = combiners_by_uid[k].predict(fcst)
                fc = combiners_uncond[k].predict(fcst)

            ensembles[k] = fc
            ensembles[f'{k}(UID)'] = fc_uid

        ensembles_df = pd.DataFrame(ensembles)
        fcst_df = pd.concat([fcst, ensembles_df], axis=1)
        fcst_df = _harmonize_keys(fcst_df)
        test = _harmonize_keys(test)
        assert_ds_match(fcst_df, test)
        fcst_df = fcst_df.merge(test, on=['unique_id', 'ds'])

        fcst_df.to_csv(out_fp, index=False)
