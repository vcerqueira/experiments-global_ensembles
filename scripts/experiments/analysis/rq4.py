import re

import pandas as pd
import plotnine as p9
import matplotlib

# matplotlib.use('TkAgg')
matplotlib.use('agg')

from utils.workflows.plots import Plots
from utils.workflows.results import ResultAnalysis

pd.set_option('display.max_columns', None)

# RQ3: impact of forecasting horizon, worst-case scenarios

ENSEMBLES = ['MLpol', 'MLewa', 'ADE', 'BestOnTrain', 'EqAverage', 'Windowing', 'BLAST']
ENSEMBLES_UID = [f'{x}(uid)' for x in ENSEMBLES]

BASE_BENCHMARKS = ['NHITS', 'SNaive']

METADATA = ['Horizon', 'Data', 'Frequency']
METADATA_UID = METADATA + ['unique_id']

scores_uid = pd.read_csv('assets/scores_uid.csv').set_index(['Data', 'Frequency'])
scores_1h = pd.read_csv('assets/scores_1h.csv').set_index(['Data', 'Frequency'])
scores_fh = pd.read_csv('assets/scores_fh.csv').set_index(['Data', 'Frequency'])

# + --- expected shortfall

es_df = ResultAnalysis.exp_shortfall(scores_uid[ENSEMBLES + BASE_BENCHMARKS], 0.95).sort_values()

es_df = es_df.reset_index()
es_df.columns = ['Model', 'SMAPE(ES)']
es_df['Model'] = pd.Categorical(es_df['Model'].values.tolist(),
                                categories=es_df['Model'].values.tolist())

plot = \
    p9.ggplot(data=es_df,
              mapping=p9.aes(x='Model',
                             y='SMAPE(ES)')) + \
    p9.geom_bar(position='dodge',
                stat='identity',
                width=0.9,
                fill='#8d021f') + \
    Plots.get_theme() + \
    p9.theme(axis_title_y=p9.element_text(size=14),
             axis_text=p9.element_text(size=13)) + \
    p9.labs(x='', y='SMAPE')

plot.save('assets/outputs/plot4.pdf', width=12, height=5)

# + --- horizon

h1 = scores_1h[ENSEMBLES + BASE_BENCHMARKS].mean()
hf = scores_fh[ENSEMBLES + BASE_BENCHMARKS].mean()

h1 = h1.reset_index()
h1.columns = ['Model', 'SMAPE']
h1['Type'] = 'One-step ahead'
hf = hf.reset_index()
hf.columns = ['Model', 'SMAPE']
hf['Type'] = 'Multi-step ahead'

horizon_df = pd.concat([h1, hf])
horizon_df = horizon_df.melt(['Type', 'Model']).drop(columns='variable')

plot = p9.ggplot(data=horizon_df,
                 mapping=p9.aes(x='Model',
                                y='value',
                                fill='Type')) + \
       p9.facet_grid('~Type') + \
       p9.geom_bar(position='dodge',
                   stat='identity',
                   width=0.9) + \
       Plots.get_theme() + \
       p9.theme(axis_text_x=p9.element_text(angle=60, size=13),
                strip_text=p9.element_text(size=13)) + \
       p9.labs(x='', y='SMAPE') + \
       p9.guides(fill=False) + \
       p9.scale_fill_manual(values=['#23395d', '#8da9c4'])

# '#152238'

plot.save('assets/outputs/plot5.pdf', width=12, height=5)
