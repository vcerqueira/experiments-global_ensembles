import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# matplotlib.use('TkAgg')
matplotlib.use('agg')

from src.plots import Plots
from src.result_analysis import ResultAnalysis

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

Plots.get_theme()
fig, ax = plt.subplots(figsize=(12, 5))
sns.barplot(
    data=es_df,
    x='Model',
    y='SMAPE(ES)',
    color='#8d021f',
    width=0.9,
    ax=ax,
)
ax.set_xlabel('')
ax.set_ylabel('SMAPE', fontsize=14)
ax.tick_params(labelsize=13)
fig.savefig('assets/outputs/plot4.pdf', bbox_inches='tight')
plt.close(fig)

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

g = sns.catplot(
    data=horizon_df,
    x='Model',
    y='value',
    hue='Type',
    col='Type',
    kind='bar',
    palette=['#23395d', '#8da9c4'],
    width=0.9,
    legend=False,
    errorbar=None,
    height=5,
    aspect=1.2,
)
g.set_axis_labels('', 'SMAPE')
g.set_xticklabels(rotation=60, fontsize=13)
g.set_titles('{col_name}', size=13)
g.fig.set_size_inches(12, 5)
g.tight_layout()
g.savefig('assets/outputs/plot5.pdf', bbox_inches='tight')
plt.close(g.fig)

# '#152238'
