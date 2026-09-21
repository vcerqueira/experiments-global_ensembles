from pathlib import Path

import pandas as pd

from src.config import ENSEMBLES
from src.utils import to_latex_tab

WEIGHT_BY_UID = True
WEIGHTS = 'fitted'
SCORES_FP = Path(f'assets/scores_uid,weights-{WEIGHTS}.csv')

ASPECTS = {
    'stationarity': ['Stationary', 'Non-stationary'],
    'heteroskedasticity': ['Homoskedastic', 'Heteroskedastic'],
    'seasonality': ['Seasonal', 'Non-seasonal'],
}

if WEIGHT_BY_UID:
    models = [f'{x}(UID)' for x in ENSEMBLES]
else:
    models = list(ENSEMBLES)

cv = pd.read_csv(SCORES_FP)
cv = cv.loc[cv['Horizon'] == 'overall']

col_order = cv[models].mean().sort_values().index.tolist()

rows = []
for aspect, levels in ASPECTS.items():
    part = cv.groupby(aspect)[models].mean()
    part = part.reindex([lvl for lvl in levels if lvl in part.index])
    rows.append(part)
table = pd.concat(rows).loc[:, col_order]
if WEIGHT_BY_UID:
    table.columns = [c.removesuffix('(UID)') for c in table.columns]
table = table.T

text_tab = to_latex_tab(
    table,
    round_to_n=3,
    rotate_cols=True,
    caption='Average MASE by series aspect.',
    label='tab:mase_aspects',
    axis=0,
)
print(text_tab)
