import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# matplotlib.use('TkAgg')
matplotlib.use('agg')

from src.plots import Plots

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

Plots.get_theme()
fig, ax = plt.subplots(figsize=(12, 5))
sns.violinplot(
    data=base_ranks,
    x='variable',
    y='value',
    color='#8d021f',
    inner=None,
    ax=ax,
)
sns.pointplot(
    data=base_ranks,
    x='variable',
    y='value',
    color='yellow',
    errorbar=('ci', 95),
    ax=ax,
)
ax.set_xlabel('')
ax.set_ylabel('Rank', fontsize=14)
ax.tick_params(labelsize=14)
fig.savefig('assets/outputs/plot0.pdf', bbox_inches='tight')
plt.close(fig)
