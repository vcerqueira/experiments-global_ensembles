import warnings
from pathlib import Path
from statsforecast import StatsForecast
from statsforecast.models import SeasonalNaive

from neuralforecast import NeuralForecast

from src.loaders import ChronosDataset, LongHorizonDatasetR
from src.neuralnets import BaseModelsConfig

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

train, _ = ChronosDataset.time_wise_split(df, horizon)

# ---- model setup
if __name__ == '__main__':
    print(RESULTS_PATH.absolute())
    models_sf = [SeasonalNaive(season_length=seas_len)]
    models_nf = BaseModelsConfig.get_nf_models(horizon=horizon,
                                               try_mps=False,
                                               input_size=n_lags,
                                               limit_epochs=False)

    sf = StatsForecast(models=models_sf, freq=freq, n_jobs=1, )
    nf = NeuralForecast(models=models_nf, freq=freq)

    # ---- cv forecasts
    n_windows = train['unique_id'].value_counts().min() - n_lags - horizon
    n_windows = int(n_windows // 2)

    # h=2 hack
    fcst_cv_sf = sf.cross_validation(df=train, n_windows=n_windows, step_size=1, h=2)
    fcst_cv_sf = fcst_cv_sf.reset_index()
    fcst_cv_sf = fcst_cv_sf.groupby(['unique_id', 'cutoff']).head(1).drop(columns='cutoff')
    fcst_cv_sf = fcst_cv_sf.reset_index(drop=True)

    # todo use nf.predict_insample(step_size=1) ???
    fcst_cv_nf = nf.cross_validation(df=train,
                                     n_windows=n_windows,
                                     step_size=1)
    fcst_cv_nf = fcst_cv_nf.groupby(['unique_id', 'cutoff']).head(1).drop(columns='cutoff')
    fcst_cv_nf = fcst_cv_nf.reset_index(drop=True)

    fcst_cv = fcst_cv_nf.merge(fcst_cv_sf.drop(columns='y'), on=['unique_id', 'ds'])

    fcst_cv.to_csv(RESULTS_PATH / f'{target},insample-base-fcst.csv', index=False)
