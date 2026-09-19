# Dynamic Ensembles for Global Time Series Forecasting

Code for experiments on dynamic ensembles applied to collections of time series with global forecasting models.

The forecasting stack is [NeuralForecast](https://github.com/Nixtla/neuralforecast) / [StatsForecast](https://github.com/Nixtla/statsforecast). Combination and evaluation use [metaforecast](https://github.com/vcerqueira/metaforecast).

## Overview

The study asks:

1. Do dynamic ensembles of global models improve accuracy relative to individual global models?
2. Which dynamic ensemble works best in this setting?
3. Does the gain change with the forecast horizon, series aspects, or difficult series?
4. Should combination weights be estimated on the whole dataset or separately for each series?
5. Should those weights be fitted on in-sample (training) loss or on cross-validation forecasts?

## Data

Datasets are loaded at run time (Chronos / Hugging Face and, for long-horizon groups, datasetsforecast). They are not stored in the repo.


## Repository layout

- `src/config.py` — datasets, device, dry-run, trim ratio, `USE_TRAINING_LOSS` (fitted vs CV weights)
- `src/loaders/` — Chronos and long-horizon loaders
- `src/neuralnets.py` — Auto-model search and best-config reconstruction
- `src/plots.py`, `src/utils.py` — plotting theme and LaTeX tables
- `scripts/experiments/run-cv.py` — fit base models; write test, fitted, and CV forecasts
- `scripts/experiments/run-ensembles.py` — fit combiners and write ensemble forecasts
- `scripts/experiments/run-evaluation.py` — MASE per series, aspect tags, cumulative horizons
- `scripts/experiments/run_outputs/` — tables and figures from the score file
- `assets/results_cv/` — base-model forecasts
- `assets/results/` — ensemble forecasts
- `assets/scores_uid,weights-{fitted|cv}.csv` — evaluation table
- `assets/outputs/` — plots

## Installation

Python 3.11+. From the repository root:

```bash
pip install -e .
```

Hardware is selected in `src/config.py` (`USE_MPS`, `USE_CUDA`) or via those environment variables.

## Reproducing the experiments

Run the three stages in order, from the repository root:

```bash
python scripts/experiments/run-cv.py
python scripts/experiments/run-ensembles.py
python scripts/experiments/run-evaluation.py
```

1. **CV** trains the Auto models and SeasonalNaive, then writes `assets/results_cv/{dataset},base-fcst.csv`, `base-fitted-fcst.csv`, and `base-cv-fcst.csv`.
2. **Ensembles** combine those forecasts. `USE_TRAINING_LOSS=True` weights on fitted loss (`weights-fitted`); `False` uses CV forecasts (`weights-cv`). Output: `assets/results/{dataset},ensemble-fcst,weights-{fitted|cv}.csv`.
3. **Evaluation** reloads the training split (needed for MASE), scores every model per series, and writes `assets/scores_uid,weights-{fitted|cv}.csv`. Each row is one dataset × series × horizon, with stationarity, heteroskedasticity, and seasonality labels.

Then run the scripts in `scripts/experiments/run_outputs/`:

| Script | Output |
| --- | --- |
| `pre-rank-dist.py` | Rank distribution of the base models |
| `mase_scores.py` | MASE table by dataset, plus top-*k* dataset wins |
| `aspects.py` | MASE by stationarity, heteroskedasticity, seasonality |
| `uid_effect.py` | Dataset-wide vs per-series (`UID`) weights |
| `horizon.py` | Overall vs first vs last horizon |
| `metadata-source.py` | Fitted vs CV combination weights |

Set `WEIGHT_BY_UID`, `WEIGHTS` (`fitted` / `cv`), `K`, and `PLOT_EXTENSION` inside those scripts as needed.
