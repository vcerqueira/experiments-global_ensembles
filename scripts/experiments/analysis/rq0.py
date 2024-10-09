import pandas as pd
import plotnine as p9
import matplotlib

# matplotlib.use('TkAgg')
matplotlib.use('agg')

from utils.workflows.plots import Plots

pd.set_option('display.max_columns', None)

# RQ0: EDA on base models

ENSEMBLES = ['MLpol', 'MLewa', 'ADE', 'LossOnTrain', 'BestOnTrain', 'EqAverage', 'Windowing', 'BLAST']
ENSEMBLES_UID = [f'{x}(uid)' for x in ENSEMBLES]
BASE_MODELS = ['NBEATS', 'NHITS', 'MLP', 'MLP(3L)', 'LSTM', 'GRU',
               'DLSTM', 'DGRU',
               'TCN', 'TiDE', 'SNaive']

METADATA = ['Horizon', 'Data', 'Frequency']
METADATA_UID = METADATA + ['unique_id']

scores_df = pd.read_csv('assets/scores_avg.csv')
scores_uid = pd.read_csv('assets/scores_uid.csv')

base_ranks = scores_df[BASE_MODELS].rank(axis=1).melt()
avg_rank = base_ranks.groupby('variable').mean().reset_index()
ord = avg_rank.sort_values('value')['variable'].values

# +----------------

base_ranks['variable'] = pd.Categorical(base_ranks['variable'], categories=ord)

plot = \
    p9.ggplot(data=base_ranks,
              mapping=p9.aes(x='variable', y='value')) + \
    p9.geom_violin(fill='#8d021f') + \
    p9.stat_summary(fun_data='mean_cl_boot', color="yellow") + \
    Plots.get_theme() + \
    p9.theme(axis_title_y=p9.element_text(size=14),
             axis_text=p9.element_text(size=14)) + \
    p9.labs(x='', y='Rank')

plot.save('assets/outputs/plot0.pdf', width=12, height=5)
