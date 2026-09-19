from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

matplotlib.use('agg')

from src.config import ENSEMBLES

WEIGHT_BY_UID = True
PLOT_EXTENSION = 'png'
OUTPUT_DIR = Path('assets/outputs')
SOURCES = {
    'Fitted': Path('assets/scores_uid,weights-fitted.csv'),
    'CV': Path('assets/scores_uid,weights-cv.csv'),
}

if WEIGHT_BY_UID:
    models = [f'{name}(UID)' for name in ENSEMBLES]
else:
    models = list(ENSEMBLES)

parts = []
for label, fp in SOURCES.items():
    if not fp.exists():
        print(f'Skipping {label}: missing {fp}')
        continue

    scores = pd.read_csv(fp)
    scores = scores.loc[scores['Horizon'] == 'overall', models].copy()
    if WEIGHT_BY_UID:
        scores.columns = ENSEMBLES
    long = scores.melt(var_name='Model', value_name='MASE')
    long['Metadata'] = label
    parts.append(long)

if not parts:
    raise SystemExit('No score files found.')

plot_df = pd.concat(parts, ignore_index=True)

order_src = plot_df.loc[plot_df['Metadata'] == 'Fitted']
if order_src.empty:
    order_src = plot_df
order = order_src.groupby('Model')['MASE'].mean().sort_values().index.tolist()
plot_df['Model'] = pd.Categorical(plot_df['Model'], categories=order)

sns.set_theme(
    style='whitegrid',
    rc={
        'font.family': 'serif',
        'font.serif': ['Georgia', 'Palatino', 'Times New Roman'],
    },
)
present = [label for label in SOURCES if label in set(plot_df['Metadata'])]
fig, ax = plt.subplots(figsize=(12, 5))
sns.barplot(
    data=plot_df,
    x='Model',
    y='MASE',
    hue='Metadata',
    hue_order=present,
    palette=['#7a1f2b', '#5b7c99'][:len(present)],
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
fig.savefig(OUTPUT_DIR / f'metadata-source.{PLOT_EXTENSION}', bbox_inches='tight')
plt.close(fig)
