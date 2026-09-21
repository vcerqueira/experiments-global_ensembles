from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

matplotlib.use('agg')

from src.config import ENSEMBLES
from src.utils import to_latex_tab

PLOT_EXTENSION = 'png'
SCORES_FP = Path('assets/scores_uid,weights-fitted.csv')
OUTPUT_DIR = Path('assets/outputs')

cv = pd.read_csv(SCORES_FP)
cv = cv.loc[cv['Horizon'] == 'overall']

ensembles_uid = [f'{name}(UID)' for name in ENSEMBLES]

overall = cv[ENSEMBLES].melt(var_name='Model', value_name='MASE')
overall['Weighting'] = 'Overall'

uid = cv[ensembles_uid].copy()
uid.columns = ENSEMBLES
uid = uid.melt(var_name='Model', value_name='MASE')
uid['Weighting'] = 'By time series'

plot_df = pd.concat([overall, uid], ignore_index=True)

order = overall.groupby('Model')['MASE'].mean().sort_values().index.tolist()
plot_df['Model'] = pd.Categorical(plot_df['Model'], categories=order)
weighting = ['Overall', 'By time series']

table = (
    plot_df.groupby(['Model', 'Weighting'], observed=True)['MASE']
    .mean()
    .unstack('Weighting')
    .reindex(index=order, columns=weighting)
)
print(
    to_latex_tab(
        table,
        round_to_n=3,
        caption='Average MASE by ensemble method and weighting scheme.',
        label='tab:uid_effect',
        mark_second=False,
    )
)

sns.set_theme(
    style='whitegrid',
    rc={
        'font.family': 'serif',
        'font.serif': ['Georgia', 'Palatino', 'Times New Roman'],
    },
)
fig, ax = plt.subplots(figsize=(12, 5))
sns.barplot(
    data=plot_df,
    x='Model',
    y='MASE',
    hue='Weighting',
    hue_order=weighting,
    palette=['#7a1f2b', '#5b7c99'],
    errorbar=None,
    width=0.8,
    ax=ax,
)
ax.set_xlabel('')
ax.set_ylabel('Average MASE', fontsize=16)
ax.tick_params(axis='x', labelsize=16, rotation=45)
ax.tick_params(axis='y', labelsize=16)
ax.legend(title='', fontsize=13)
sns.despine(left=True, bottom=True)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / f'uid_effect.{PLOT_EXTENSION}', bbox_inches='tight')
plt.close(fig)
