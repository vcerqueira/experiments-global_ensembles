from pathlib import Path

import pandas as pd

from src.config import ENSEMBLES
from src.utils import to_latex_tab

WEIGHT_BY_UID = True
WEIGHTS = 'fitted'
K = 3
SCORES_FP = Path(f'assets/scores_uid,weights-{WEIGHTS}.csv')

if WEIGHT_BY_UID:
    models = [f'{x}(UID)' for x in ENSEMBLES]
else:
    models = list(ENSEMBLES)

cv = pd.read_csv(SCORES_FP)
cv = cv.loc[cv['Horizon'] == 'overall']

scores = cv.groupby(['Data', 'Frequency'], sort=True)[models].mean()
overall = cv[models].mean().to_frame().T
overall.index = pd.MultiIndex.from_tuples([('All', '')], names=scores.index.names)

col_order = overall.iloc[0].sort_values().index.tolist()
topk = scores.rank(axis=1, method='min', ascending=True).le(K).sum()
topk = topk.to_frame().T
topk.index = pd.MultiIndex.from_tuples([(f'Top-{K}', '')], names=scores.index.names)

table = pd.concat([scores, overall, topk]).loc[:, col_order]
if WEIGHT_BY_UID:
    table.columns = [c.removesuffix('(UID)') for c in table.columns]

text_tab = to_latex_tab(
    table,
    round_to_n=3,
    rotate_cols=True,
    caption=f'Average MASE by dataset and number of datasets in the top {K}.',
    label='tab:mase_scores',
)
print(text_tab)
