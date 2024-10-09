import re

import pandas as pd
import plotnine as p9
import matplotlib

# matplotlib.use('TkAgg')
matplotlib.use('agg')

from utils.workflows.plots import Plots

pd.set_option('display.max_columns', None)

# RQ3: ensemble by uid or overall?

ENSEMBLES = ['MLpol', 'MLewa', 'ADE', 'BestOnTrain', 'EqAverage', 'Windowing', 'BLAST']
ENSEMBLES_UID = [f'{x}(uid)' for x in ENSEMBLES]

BASE_BENCHMARKS = ['NHITS', 'SNaive']

METADATA = ['Horizon', 'Data', 'Frequency']
METADATA_UID = METADATA + ['unique_id']

scores_df = pd.read_csv('assets/scores_avg.csv').set_index(['Data', 'Frequency'])

# + --- avg score differences

avg_unc = scores_df[ENSEMBLES]
avg_uid = scores_df[ENSEMBLES_UID]
avg_uid.columns = [re.sub('\(uid\)', '', x) for x in avg_uid.columns]

delta = avg_unc.mean() - avg_uid.mean()

delta = delta.reset_index()
delta.columns = ['Model', 'SMAPE difference']
delta['Model'] = pd.Categorical(delta['Model'].values.tolist(),
                                categories=delta['Model'].values.tolist())

plot = \
    p9.ggplot(data=delta,
              mapping=p9.aes(x='Model',
                             y='SMAPE difference')) + \
    p9.geom_bar(position='dodge',
                stat='identity',
                width=0.9,
                fill='#8d021f') + \
    Plots.get_theme() + \
    p9.theme(axis_title_y=p9.element_text(size=14),
             axis_text=p9.element_text(size=13)) + \
    p9.labs(x='', y='SMAPE difference')

plot.save('assets/outputs/plot3.pdf', width=12, height=5)
