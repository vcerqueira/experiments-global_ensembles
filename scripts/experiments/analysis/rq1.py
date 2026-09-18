import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# matplotlib.use('TkAgg')
matplotlib.use('agg')

from src.plots import Plots

pd.set_option('display.max_columns', None)

# RQ2: Do dynamic ensembles applied with global forecasting models
# improve forecasting accuracy relative to individual (global) models?

ENSEMBLES = ['MLpol', 'MLewa', 'ADE', 'BestOnTrain', 'EqAverage', 'Windowing', 'BLAST']
ENSEMBLES_UID = [f'{x}(uid)' for x in ENSEMBLES]

BASE_BENCHMARKS = ['NHITS', 'SNaive']

METADATA = ['Horizon', 'Data', 'Frequency']
METADATA_UID = METADATA + ['unique_id']

scores_df = pd.read_csv('assets/scores_avg.csv').set_index(['Data', 'Frequency'])
scores_uid = pd.read_csv('assets/scores_uid.csv').set_index(['Data', 'Frequency'])

# + --- avg score

avg = scores_df[ENSEMBLES + BASE_BENCHMARKS].mean().sort_values()
avg = avg.reset_index()
avg.columns = ['Model', 'SMAPE']
avg['Model'] = pd.Categorical(avg['Model'].values.tolist(),
                              categories=avg['Model'].values.tolist())

Plots.get_theme()
fig, ax = plt.subplots(figsize=(12, 5))
sns.barplot(
    data=avg,
    x='Model',
    y='SMAPE',
    color='#8d021f',
    width=0.9,
    ax=ax,
)
ax.set_xlabel('')
ax.set_ylabel('SMAPE', fontsize=14)
ax.tick_params(labelsize=13)
fig.savefig('assets/outputs/plot1.pdf', bbox_inches='tight')
plt.close(fig)

# + --- avg rank

ens_ranks = scores_df[ENSEMBLES + BASE_BENCHMARKS].rank(axis=1).melt()
avg_rank = ens_ranks.groupby('variable').mean().reset_index()
ord = avg_rank.sort_values('value')['variable'].values
ens_ranks['variable'] = pd.Categorical(ens_ranks['variable'], categories=ord)

fig, ax = plt.subplots(figsize=(12, 5))
sns.violinplot(
    data=ens_ranks,
    x='variable',
    y='value',
    color='#8d021f',
    inner=None,
    ax=ax,
)
sns.pointplot(
    data=ens_ranks,
    x='variable',
    y='value',
    color='yellow',
    errorbar=('ci', 95),
    ax=ax,
)
ax.set_xlabel('')
ax.set_ylabel('Rank', fontsize=14)
ax.tick_params(labelsize=13)
fig.savefig('assets/outputs/plot2.pdf', bbox_inches='tight')
plt.close(fig)

# scores_df[ENSEMBLES + BASE_BENCHMARKS].rank(axis=1).mean().sort_values()
# scores_df[ENSEMBLES + BASE_BENCHMARKS].mean().sort_values()
# scores_df[ENSEMBLES + BASE_BENCHMARKS].round(4)
# scores_df[ENSEMBLES + BASE_BENCHMARKS].round(4).mean().sort_values()
