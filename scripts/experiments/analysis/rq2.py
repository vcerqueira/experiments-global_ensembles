import pandas as pd
import plotnine as p9
import matplotlib

# matplotlib.use('TkAgg')
matplotlib.use('agg')

from utils.workflows.plots import Plots

pd.set_option('display.max_columns', None)

# RQ2: what is the best approach

ENSEMBLES = ['MLpol', 'MLewa', 'ADE', 'BestOnTrain', 'EqAverage', 'Windowing', 'BLAST']
ENSEMBLES_UID = [f'{x}(uid)' for x in ENSEMBLES]

BASE_BENCHMARKS = ['NHITS', 'SNaive']

METADATA = ['Horizon', 'Data', 'Frequency']
METADATA_UID = METADATA + ['unique_id']

scores_df = pd.read_csv('assets/scores_avg.csv').set_index(['Data', 'Frequency'])

# + --- avg score

ord = scores_df[ENSEMBLES + BASE_BENCHMARKS].mean().sort_values().sort_values().index.tolist()

res = scores_df[ENSEMBLES + BASE_BENCHMARKS].round(4)
res = res.astype(str)
res.index = ['\_'.join(x) for x in res.index]

# res.index = [f'\\rotatebox{{90}}{{{x}}}' for x in res.index]

res = res.loc[:, ord]
res = res.sort_index()

res.columns = [f'\\rotatebox{{90}}{{{x}}}' for x in res.columns]

annotated_res = []
for i, r in res.iterrows():
    top_2 = r.sort_values().unique()[:2]
    if len(top_2) < 2:
        raise ValueError('only one score')

    best1 = r[r == top_2[0]].values[0]
    best2 = r[r == top_2[1]].values[0]

    r[r == top_2[0]] = f'\\textbf{{{best1}}}'
    r[r == top_2[1]] = f'\\underline{{{best2}}}'

    annotated_res.append(r)

annotated_res = pd.DataFrame(annotated_res)

text_tab = annotated_res.to_latex(caption='CAPTION', label='tab:scores_by_ds')

print(text_tab)
