import pandas as pd
import plotnine as p9
import matplotlib

# matplotlib.use('TkAgg')
matplotlib.use('agg')

from utils.workflows.plots import Plots

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

plot = \
    p9.ggplot(data=avg,
              mapping=p9.aes(x='Model',
                             y='SMAPE')) + \
    p9.geom_bar(position='dodge',
                stat='identity',
                width=0.9,
                fill='#8d021f') + \
    Plots.get_theme() + \
    p9.theme(axis_title_y=p9.element_text(size=14),
             axis_text=p9.element_text(size=13)) + \
    p9.labs(x='', y='SMAPE')

plot.save('assets/outputs/plot1.pdf', width=12, height=5)

# + --- avg rank

ens_ranks = scores_df[ENSEMBLES + BASE_BENCHMARKS].rank(axis=1).melt()
avg_rank = ens_ranks.groupby('variable').mean().reset_index()
ord = avg_rank.sort_values('value')['variable'].values
ens_ranks['variable'] = pd.Categorical(ens_ranks['variable'], categories=ord)

plot = \
    p9.ggplot(data=ens_ranks,
              mapping=p9.aes(x='variable', y='value')) + \
    p9.geom_violin(fill='#8d021f') + \
    p9.stat_summary(fun_data='mean_cl_boot', color="yellow") + \
    Plots.get_theme() + \
    p9.theme(axis_title_y=p9.element_text(size=14),
             axis_text=p9.element_text(size=13)) + \
    p9.labs(x='', y='Rank')

plot.save('assets/outputs/plot2.pdf', width=12, height=5)

# scores_df[ENSEMBLES + BASE_BENCHMARKS].rank(axis=1).mean().sort_values()
# scores_df[ENSEMBLES + BASE_BENCHMARKS].mean().sort_values()
# scores_df[ENSEMBLES + BASE_BENCHMARKS].round(4)
# scores_df[ENSEMBLES + BASE_BENCHMARKS].round(4).mean().sort_values()
