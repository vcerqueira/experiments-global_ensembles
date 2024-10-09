# Dynamic Ensembles for Global Time Series Forecasting

This repository contains the code and data for experiments on dynamic 
ensembles for global time series forecasting.

## Overview

We investigate the application of dynamic ensembles to 
collections of time series using global forecasting models. 
Our study explores several research questions:

1. Do dynamic ensembles applied with global forecasting models improve forecasting accuracy relative to individual (global) models?
2. What is the best dynamic ensemble approach in this context?
3. Does the impact of dynamic ensembles vary under different forecasting conditions, such as the forecasting horizon and worst-case scenarios?
4. How should dynamic ensembles compute the weights of each model - by each time series individually or by the overall dataset?

## Data

WIP

The datasets are not included in this repository due to size constraints.

## Methods

WIP

## Code Structure

The experiments are based on the Nixtla's ecosystem and metaforecast

- `utils/`: Contains utility functions for data loading and preprocessing
- `experiments/run`: Scripts to run experiments
- `experiments/analysis/`: Scripts to analyse results
- `assets/`: Main outputs 

## Requirements

Required Python packages are listed in `requirements.txt`. Install them using:

pip install -r requirements.txt

## Reproducing experiments

To reproduce the experiments, follow:

- `experiments/run`:


Feel free to get in touch