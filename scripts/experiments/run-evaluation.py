import warnings
from functools import partial
from pathlib import Path

import pandas as pd
from utilsforecast.losses import mase

from metaforecast.evaluation.aspects import (
    DifferencingTests,
    Heteroskedasticity,
    ModelRadar,
)

from src.loaders import ChronosDataset, LongHorizonDatasetR
from src.config import DATASETS, LH_DATASETS, USE_TRAINING_LOSS

warnings.filterwarnings("ignore")

ENSEMBLE_RESULTS_PATH = Path(__file__).resolve().parents[2] / "assets" / "results"
# ENSEMBLE_RESULTS_PATH = Path("assets/results")
SCORES_PATH = Path(__file__).resolve().parents[2] / "assets"
# SCORES_PATH = Path("assets/")

ASPECT_COLS = ["stationarity", "heteroskedasticity"]
KEY_COLS = ["Dataset", "Data", "Frequency", "unique_id", "Horizon"] + ASPECT_COLS


def _harmonize_keys(df):
    df = df.copy()
    df["unique_id"] = df["unique_id"].astype(str)
    df["ds"] = pd.to_datetime(df["ds"]).astype("datetime64[ns]")
    return df


def load_dataset(target):
    if target in LH_DATASETS:
        _, horizon, n_lags, _, _ = LongHorizonDatasetR.load_everything(target, resample_to="D")
        df, horizon, n_lags, freq, seas_len = LongHorizonDatasetR.load_everything(
            target,
            min_n_instances=2 * (n_lags + horizon),
            resample_to="D",
        )
    else:
        _, horizon, n_lags, _, _ = ChronosDataset.load_everything(target)
        df, horizon, _, freq, seas_len = ChronosDataset.load_everything(
            target,
            min_n_instances=2 * (n_lags + horizon),
        )

    train, test = ChronosDataset.time_wise_split(df, horizon)
    return train, test, horizon, n_lags, freq, seas_len


def dataset_labels(target):
    if target in LH_DATASETS:
        return target, "Daily"

    *data_parts, freq = target.removeprefix("monash_").split("_")
    return "_".join(data_parts), freq.capitalize()


def tag_series_aspects(cv_df, train):
    stationarity_labels = {}
    het_labels = {}
    for uid, uid_df in train.groupby("unique_id"):
        y = uid_df["y"]
        try:
            ndiffs = DifferencingTests.ndiffs(y, test="kpss")
            stationarity_labels[uid] = "Stationary" if ndiffs == 0 else "Non-stationary"
        except Exception:
            stationarity_labels[uid] = "Unknown"
        try:
            bp_pvalue = Heteroskedasticity.het_tests(y, test="Breusch-Pagan")
            het_labels[uid] = "Heteroskedastic" if bp_pvalue < 0.05 else "Homoskedastic"
        except Exception:
            het_labels[uid] = "Unknown"

    cv_df = cv_df.copy()
    cv_df["stationarity"] = cv_df["unique_id"].map(stationarity_labels).fillna("Unknown")
    cv_df["heteroskedasticity"] = cv_df["unique_id"].map(het_labels).fillna("Unknown")
    return cv_df


def uid_scores_with_horizons(radar):
    overall = radar.evaluate_by_uid().assign(Horizon="overall")
    h_col = radar.COLUMNS["horizon"]
    parts = [overall]
    for h in sorted(radar.cv_df[h_col].unique()):
        sl = radar.evaluate(radar.cv_df.loc[radar.cv_df[h_col] <= h], keep_uids=True)
        parts.append(sl.assign(Horizon=int(h)))
    return pd.concat(parts)


def evaluate_forecasts(fcst, train, seas_len):
    fcst = tag_series_aspects(fcst, train)
    radar = ModelRadar(
        cv_df=fcst,
        metrics=[partial(mase, seasonality=seas_len)],
        train_df=train,
    )
    scores = uid_scores_with_horizons(radar)
    scores = scores.rename_axis("unique_id").reset_index()
    aspects = fcst[["unique_id"] + ASPECT_COLS].drop_duplicates()
    return scores.merge(aspects, on="unique_id", how="left")


if __name__ == "__main__":
    SCORES_PATH.mkdir(parents=True, exist_ok=True)
    print(ENSEMBLE_RESULTS_PATH.absolute())

    scores_by_uid = []
    for target in DATASETS:
        print(f"Evaluating {target}")
        weight_tag = "fitted" if USE_TRAINING_LOSS else "cv"
        fcst_fp = ENSEMBLE_RESULTS_PATH / f"{target},ensemble-fcst,weights-{weight_tag}.csv"

        if not fcst_fp.exists():
            print(f"Skipping {target}: missing ensemble forecasts")
            continue

        train, _, _, _, _, seas_len = load_dataset(target)
        train = _harmonize_keys(train)

        fcst = pd.read_csv(fcst_fp, parse_dates=["ds"])
        fcst = _harmonize_keys(fcst)

        data, frequency = dataset_labels(target)
        uid = evaluate_forecasts(fcst, train, seas_len)
        uid["Dataset"] = target
        uid["Data"] = data
        uid["Frequency"] = frequency
        scores_by_uid.append(uid)

        n_uid = uid["unique_id"].nunique()
        n_h = uid.loc[uid["Horizon"] != "overall", "Horizon"].nunique()
        print(f"{n_uid} series, {n_h} cumulative horizons")

    if not scores_by_uid:
        raise SystemExit("No ensemble forecasts found to evaluate.")

    scores_uid_df = pd.concat(scores_by_uid, axis=0, ignore_index=True)
    model_cols = [c for c in scores_uid_df.columns if c not in KEY_COLS]
    scores_uid_df = scores_uid_df[KEY_COLS + model_cols]
    scores_uid_df.to_csv(SCORES_PATH / "scores_uid.csv", index=False)
