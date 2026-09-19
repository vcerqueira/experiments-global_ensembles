import re

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# matplotlib.use('TkAgg')
matplotlib.use('agg')

from src.plots import Plots

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
avg_uid.columns = [re.sub(r'\(uid\)', '', x) for x in avg_uid.columns]

delta = avg_unc.mean() - avg_uid.mean()

delta = delta.reset_index()
delta.columns = ['Model', 'SMAPE difference']
delta['Model'] = pd.Categorical(delta['Model'].values.tolist(),
                                categories=delta['Model'].values.tolist())

Plots.get_theme()
fig, ax = plt.subplots(figsize=(12, 5))
sns.barplot(
    data=delta,
    x='Model',
    y='SMAPE difference',
    color='#8d021f',
    width=0.9,
    ax=ax,
)
ax.set_xlabel('')
ax.set_ylabel('SMAPE difference', fontsize=14)
ax.tick_params(labelsize=13)
fig.savefig('assets/outputs/plot3.pdf', bbox_inches='tight')
plt.close(fig)
