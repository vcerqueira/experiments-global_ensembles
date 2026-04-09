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

    sf.fit(df=train)
    nf.fit(df=train)

    fcst_sf = sf.predict(h=horizon)
    fcst_ml = nf.predict()

    fcst = fcst_ml.merge(fcst_sf, on=['unique_id', 'ds']).reset_index()

    fcst.to_csv(RESULTS_PATH / f'{target},base-fcst.csv', index=False)
