from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

matplotlib.use('agg')

from src.config import ENSEMBLES
from src.utils import to_latex_tab

WEIGHT_BY_UID = True
WEIGHTS = 'fitted'
PLOT_EXTENSION = 'png'
SCORES_FP = Path(f'assets/scores_uid,weights-{WEIGHTS}.csv')
OUTPUT_DIR = Path('assets/outputs')

if WEIGHT_BY_UID:
    models = [f'{name}(UID)' for name in ENSEMBLES]
else:
    models = list(ENSEMBLES)

cv = pd.read_csv(SCORES_FP)
h = pd.to_numeric(cv['Horizon'], errors='coerce')
cv = cv.assign(_h=h)

first = cv.loc[cv['_h'] == 1]
last_h = cv.groupby('Dataset')['_h'].transform('max')
last = cv.loc[cv['_h'].eq(last_h)]


def to_long(df, label):
    out = df[models].copy()
    if WEIGHT_BY_UID:
        out.columns = ENSEMBLES
    out = out.melt(var_name='Model', value_name='MASE')
    out['Horizon'] = label
    return out


plot_df = pd.concat(
    [
        to_long(first, 'First'),
        to_long(last, 'Last'),
    ],
    ignore_index=True,
)

order = (
    plot_df.loc[plot_df['Horizon'] == 'Last']
    .groupby('Model')['MASE']
    .mean()
    .sort_values()
    .index
    .tolist()
)
plot_df['Model'] = pd.Categorical(plot_df['Model'], categories=order)
horizons = ['First', 'Last']

table = (
    plot_df.groupby(['Model', 'Horizon'], observed=True)['MASE']
    .mean()
    .unstack('Horizon')
    .reindex(index=order, columns=horizons)
)
print(
    to_latex_tab(
        table,
        round_to_n=3,
        caption='Average MASE by ensemble method and forecast horizon.',
        label='tab:horizon',
        mark_second=False,
        axis=0,
    )
)

sns.set_theme(
    style='whitegrid',
    rc={
        'font.family': 'serif',
        'font.serif': ['Georgia', 'Palatino', 'Times New Roman'],
    },
)
g = sns.catplot(
    data=plot_df,
    x='Model',
    y='MASE',
    row='Horizon',
    row_order=horizons,
    kind='bar',
    color='#7a1f2b',
    errorbar=None,
    height=2.6,
    aspect=4.4,
    sharey=True,
)
g.set_axis_labels('', 'Average MASE')
g.set_titles('{row_name}')
for ax in g.axes.flat:
    ax.tick_params(axis='y', labelsize=13)
    sns.despine(ax=ax, left=True, bottom=True)
g.axes.flat[-1].tick_params(axis='x', labelsize=13)
plt.setp(g.axes.flat[-1].get_xticklabels(), rotation=45, ha='right')
g.tight_layout()
g.savefig(OUTPUT_DIR / f'horizon.{PLOT_EXTENSION}', bbox_inches='tight')
plt.close(g.fig)
