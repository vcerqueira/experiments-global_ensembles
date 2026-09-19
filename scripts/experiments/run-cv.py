import re
import warnings
from pathlib import Path
from statsforecast import StatsForecast
from statsforecast.models import SeasonalNaive

from neuralforecast import NeuralForecast

from src.loaders import ChronosDataset, LongHorizonDatasetR
from src.neuralnets import ModelsConfig
from src.config import ENGINE, LIMIT_EPOCHS, N_SAMPLES, DATASETS, LH_DATASETS

warnings.filterwarnings("ignore")

RESULTS_PATH = Path(__file__).resolve().parents[2] / "assets" / "results_cv"

if __name__ == '__main__':
    print(RESULTS_PATH.absolute())
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)

    for target in DATASETS:
        # target = 'monash_m1_monthly'
        print(f"Running {target}")
        base_fp = RESULTS_PATH / f'{target},base-fcst.csv'
        base_int_fp = RESULTS_PATH / f'{target},base-insampletrain-fcst.csv'
        base_incv_fp = RESULTS_PATH / f'{target},base-insamplecv-fcst.csv'

        if base_fp.exists() and base_int_fp.exists() and base_incv_fp.exists():
            continue

        if target in LH_DATASETS:
            _, horizon, n_lags, _, _ = LongHorizonDatasetR.load_everything(target, resample_to='D')
            df, horizon, n_lags, freq, seas_len = LongHorizonDatasetR.load_everything(target,
                                                                                      min_n_instances=2 * (
                                                                                              n_lags + horizon),
                                                                                      resample_to='D')
        else:
            _, horizon, n_lags, _, _ = ChronosDataset.load_everything(target)
            df, horizon, _, freq, seas_len = ChronosDataset.load_everything(target,
                                                                            min_n_instances=2 * (n_lags + horizon))

        train, _ = ChronosDataset.time_wise_split(df, horizon)

        models_sf = [SeasonalNaive(season_length=seas_len)]
        models_nf = ModelsConfig.get_auto_nf_models(horizon=horizon,
                                                    engine=ENGINE,
                                                    limit_epochs=LIMIT_EPOCHS,
                                                    n_samples=N_SAMPLES)

        sf = StatsForecast(models=models_sf, freq=freq, n_jobs=1, )
        nf = NeuralForecast(models=models_nf, freq=freq)

        nf.fit(train)
        sf.fit(train)

        optim_models = ModelsConfig.get_best_configs(nf)

        fcst_sf = sf.predict(h=horizon)
        fcst_nf = nf.predict()
        fcst_nf = fcst_nf.rename(columns=lambda c: re.sub(r"^Auto", "", c))

        fcst = fcst_nf.merge(fcst_sf, on=['unique_id', 'ds'])

        fcst.to_csv(base_fp, index=False)

        fcst_nf_ins = nf.predict_insample(step_size=1)
        fcst_nf_ins = fcst_nf_ins.rename(columns=lambda c: re.sub(r"^Auto", "", c))
        fcst_nf_ins = fcst_nf_ins.groupby(['unique_id', 'cutoff']).head(1).drop(columns='cutoff').reset_index(drop=True)

        fcst_nf_ins.to_csv(base_int_fp, index=False)

        # CV with best configs
        nf_cv = NeuralForecast(models=optim_models, freq=freq)
        n_windows = train['unique_id'].value_counts().min() - n_lags - horizon
        n_windows = int(n_windows // 2)

        fcst_cv_nf = nf_cv.cross_validation(df=train,
                                            n_windows=n_windows,
                                            step_size=1)
        fcst_cv_nf = fcst_cv_nf.groupby(['unique_id', 'cutoff']).head(1).drop(columns='cutoff').reset_index(drop=True)

        fcst_cv_nf.to_csv(base_incv_fp, index=False)
