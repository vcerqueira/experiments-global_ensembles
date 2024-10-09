import warnings

import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import SeasonalNaive

from neuralforecast import NeuralForecast
from neuralforecast.models import (NHITS,
                                   LSTM,
                                   GRU,
                                   NBEATS,
                                   DilatedRNN,
                                   MLP,
                                   TCN,
                                   TiDE)

from metaforecast.ensembles.mlpol import MLpol
from metaforecast.ensembles.mlewa import MLewa
from metaforecast.ensembles.ade import ADE
from metaforecast.ensembles.static import LossOnTrain, BestOnTrain, EqAverage
from metaforecast.ensembles.windowing import Windowing

from utils.load_data.config import DATASETS
from utils.config import OUTPUT_DIRECTORY, TRIM_R

warnings.filterwarnings("ignore")

# ---- data loading and partitioning

ADE_LAGS = 13
data_name, group = 'M3', 'Monthly'

data_loader = DATASETS[data_name]
min_samples = data_loader.min_samples[group]
df, _, n_lags, freq_str, freq_int = data_loader.load_everything(group, min_n_instances=min_samples)

print(df['unique_id'].value_counts())
print(df.shape)

horizon = data_loader.horizons_map[group]

train, test = data_loader.train_test_split(df, horizon=horizon)

# ---- model setup

print('...Model setup')

models_sf = [SeasonalNaive(season_length=freq_int), ]

CONFIG = {
    # 'max_steps': 10,
    'input_size': n_lags,
    'h': horizon,
    'enable_checkpointing': True,
    'accelerator': 'cpu'}

models = [
    NBEATS(start_padding_enabled=True, **CONFIG, stack_types=["identity", "identity", "identity"]),
    NHITS(start_padding_enabled=True, **CONFIG),
    MLP(start_padding_enabled=True, **CONFIG),
    MLP(start_padding_enabled=True, num_layers=3, **CONFIG),
    LSTM(**CONFIG),
    GRU(**CONFIG),
    DilatedRNN(**CONFIG),
    DilatedRNN(cell_type='GRU', **CONFIG),
    TCN(**CONFIG),
    TiDE(start_padding_enabled=True, **CONFIG),
]

sf = StatsForecast(
    models=models_sf,
    freq=freq_str,
    n_jobs=1,
)

nf = NeuralForecast(models=models, freq=freq_str)

# ---- model fitting


# ---- insample forecasts
print('...CV')

n_windows = train['unique_id'].value_counts().min() - n_lags - horizon
n_windows = int(n_windows // 2)

# h=2 hack
fcst_cv_sf = sf.cross_validation(df=train, n_windows=n_windows, step_size=1, h=2)
fcst_cv_sf = fcst_cv_sf.reset_index()
fcst_cv_sf = fcst_cv_sf.groupby(['unique_id', 'cutoff']).head(1).drop(columns='cutoff')
fcst_cv_sf = fcst_cv_sf.reset_index(drop=True)

# nf.predict_insample(step_size=1)
# predict_insample() not working for nf
fcst_cv_nf = nf.cross_validation(df=train,
                                 n_windows=n_windows,
                                 step_size=1)
fcst_cv_nf = fcst_cv_nf.reset_index()
fcst_cv_nf = fcst_cv_nf.groupby(['unique_id', 'cutoff']).head(1).drop(columns='cutoff')
fcst_cv_nf = fcst_cv_nf.reset_index(drop=True)

fcst_cv = fcst_cv_nf.merge(fcst_cv_sf.drop(columns='y'), on=['unique_id', 'ds'])

# ---- fitting ensembles

print('...Model fitting')
print('......stats')

sf.fit(df=train)

print('......ML')
nf.fit(df=train)

print('...fitting ensembles')

# ade = ADE(freq=freq_str, meta_lags=list(range(1, ADE_LAGS)), trim_ratio=TRIM_R, trim_by_uid=True)

# fcst_cv
print(fcst_cv['unique_id'].value_counts())

combiners_by_uid = {
    'ADE': ADE(freq=freq_str, meta_lags=list(range(1, ADE_LAGS)), trim_ratio=TRIM_R, trim_by_uid=True),
    'MLpol': MLpol(loss_type='square', gradient=True, trim_ratio=TRIM_R, weight_by_uid=True),
    'MLewa': MLewa(loss_type='square', gradient=True, trim_ratio=TRIM_R, weight_by_uid=True),
    'LossOnTrain': LossOnTrain(trim_ratio=TRIM_R, weight_by_uid=True),
    'BestOnTrain': BestOnTrain(select_by_uid=True),
    'EqAverage': EqAverage(select_by_uid=True, trim_ratio=TRIM_R),
    'Windowing': Windowing(freq=freq_str, trim_ratio=TRIM_R, select_best=False, weight_by_uid=True),
    'BLAST': Windowing(freq=freq_str, trim_ratio=TRIM_R, select_best=True, weight_by_uid=True),
}

combiners_uncond = {
    'ADE': ADE(freq=freq_str, meta_lags=list(range(1, ADE_LAGS)), trim_ratio=TRIM_R, trim_by_uid=False),
    'MLpol': MLpol(loss_type='square', gradient=True, trim_ratio=TRIM_R, weight_by_uid=False),
    'MLewa': MLewa(loss_type='square', gradient=True, trim_ratio=TRIM_R, weight_by_uid=False),
    'LossOnTrain': LossOnTrain(trim_ratio=TRIM_R, weight_by_uid=False),
    'BestOnTrain': BestOnTrain(select_by_uid=False),
    'EqAverage': EqAverage(select_by_uid=False, trim_ratio=TRIM_R),
    'Windowing': Windowing(freq=freq_str, trim_ratio=TRIM_R, select_best=False, weight_by_uid=False),
    'BLAST': Windowing(freq=freq_str, trim_ratio=TRIM_R, select_best=True, weight_by_uid=False),
}

for k in combiners_by_uid:
    print(k)
    combiners_uncond[k].fit(fcst_cv)
    print(k)
    combiners_by_uid[k].fit(fcst_cv)

# ---- test forecasts
print('...test forecasts')

fcst_sf = sf.predict(h=horizon)
fcst_ml = nf.predict()

fcst = fcst_ml.merge(fcst_sf, on=['unique_id', 'ds']).reset_index()
print('...ensemble forecasts')

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

print('...saving results')
# ---- saving results

fcst_df.to_csv(f'{OUTPUT_DIRECTORY}/{data_name}_{group}_{horizon}.csv', index=False)
