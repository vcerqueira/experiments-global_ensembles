import re
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

matplotlib.use('agg')

from src.neuralnets import ModelsConfig

PLOT_EXTENSION = 'png'
SCORES_FP = Path('assets/scores_uid,weights-fitted.csv')
OUTPUT_DIR = Path('assets/outputs')

cv = pd.read_csv(SCORES_FP)
cv = cv.loc[cv['Horizon'] == 'overall']

base_models = [re.sub('^Auto*', '', x) for x in ModelsConfig.AUTO_MODEL_CLASSES]

base_scores = cv.loc[:, base_models + ['SeasonalNaive']]

base_ranks = base_scores.rank(axis=1).melt()
avg_rank = base_ranks.groupby('variable').mean().reset_index()
order = avg_rank.sort_values('value')['variable'].values
base_ranks['variable'] = pd.Categorical(base_ranks['variable'], categories=order)

# Plots.get_theme()
sns.set_theme(
    style="whitegrid",
    rc={
        "font.family": "serif",
        "font.serif": ["Georgia", "Palatino", "Times New Roman"],
    },
)
fig, ax = plt.subplots(figsize=(12, 5))
sns.violinplot(
    data=base_ranks,
    x='variable',
    y='value',
    color='#7a1f2b',
    inner='quart',
    cut=0,
    # linewidth=0.6,
    saturation=0.85,
    ax=ax,
)
ax.set_xlabel('', fontsize=18)
ax.set_ylabel('Rank distribution', fontsize=18)
ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)
ax.margins(y=0.06)
sns.despine(left=True, bottom=True)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / f'pre-rank-dist.{PLOT_EXTENSION}', bbox_inches='tight')
plt.close(fig)
