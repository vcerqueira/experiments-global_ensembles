from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure


class Plots:
    COLOR_MAP: dict[str, str] = {}
    ORDER: list[str] = []

    @staticmethod
    def get_group_colors(X, alt_group):
        return ['#69a765' if x in alt_group else '#ed9121' for x in X]

    @staticmethod
    def get_theme():
        sns.set_theme(
            style='whitegrid',
            rc={
                'font.family': 'serif',
                'font.serif': ['Palatino', 'Palatino Linotype', 'Times New Roman', 'DejaVu Serif'],
                'axes.facecolor': 'white',
                'figure.facecolor': 'white',
                'legend.facecolor': 'white',
                'axes.edgecolor': '#b0b0b0',
                'grid.color': '#e6e6e6',
                'grid.linewidth': 0.8,
                'axes.labelsize': 12,
                'xtick.labelsize': 9,
                'ytick.labelsize': 9,
            },
        )

    @classmethod
    def _ordered_models(cls, series: pd.Series, reverse: bool = False) -> list:
        unique = list(dict.fromkeys(series.tolist()))
        if not cls.ORDER:
            return unique
        order = cls.ORDER[::-1] if reverse else list(cls.ORDER)
        ordered = [model for model in order if model in unique]
        ordered.extend(model for model in unique if model not in ordered)
        return ordered

    @classmethod
    def _palette(cls, models: list[str]) -> Optional[dict[str, str]]:
        if not cls.COLOR_MAP:
            return None
        fallback = sns.color_palette('husl', n_colors=max(len(models), 1))
        return {
            model: cls.COLOR_MAP.get(model, fallback[i % len(fallback)])
            for i, model in enumerate(models)
        }

    @staticmethod
    def error_distribution_baseline(df: pd.DataFrame,
                                    baseline: str,
                                    thr: float) -> Figure:
        Plots.get_theme()
        fig, ax = plt.subplots()
        sns.histplot(
            df[baseline],
            bins=30,
            color='#69a765',
            edgecolor='black',
            alpha=0.95,
            ax=ax,
        )
        ax.axvline(thr, color='red', linewidth=1)
        ax.set_xlabel(f'Error distribution of {baseline}')
        ax.set_ylabel('Count')
        fig.tight_layout()
        return fig

    @classmethod
    def average_error_barplot(cls, df: pd.DataFrame) -> Figure:
        Plots.get_theme()
        df = df.sort_values('Error', ascending=False).reset_index(drop=True)
        models = df['Model'].tolist()
        df['Model'] = pd.Categorical(models, categories=models)

        fig, ax = plt.subplots()
        sns.barplot(
            data=df,
            x='Error',
            y='Model',
            hue='Model',
            palette=cls._palette(models),
            dodge=False,
            width=0.9,
            ax=ax,
            legend=False,
        )
        ax.invert_yaxis()
        ax.set_xlabel('Error across all series')
        ax.set_ylabel('')
        ax.tick_params(axis='x', labelsize=9)
        ax.yaxis.label.set_size(7)
        fig.tight_layout()
        return fig

    @classmethod
    def _faceted_model_bars(cls,
                            df: pd.DataFrame,
                            y: str,
                            facet: str,
                            ylabel: str) -> Figure:
        Plots.get_theme()
        df = df.copy()
        models = cls._ordered_models(df['Model'], reverse=True)
        df['Model'] = pd.Categorical(df['Model'], categories=models)

        g = sns.catplot(
            data=df,
            x='Model',
            y=y,
            hue='Model',
            col=facet,
            kind='bar',
            palette=cls._palette(models),
            width=0.9,
            legend=False,
            errorbar=None,
            height=4,
            aspect=0.9,
        )
        g.set_axis_labels('', ylabel)
        g.set_xticklabels(rotation=60, fontsize=7)
        g.set_titles('{col_name}', size=10)
        g.tight_layout()
        return g.fig

    @classmethod
    def average_error_by_freq(cls, df: pd.DataFrame) -> Figure:
        return cls._faceted_model_bars(df, y='Error', facet='Frequency', ylabel='SMAPE')

    @classmethod
    def average_win_rate_bar(cls, df: pd.DataFrame) -> Figure:
        Plots.get_theme()
        df = df.copy()
        models = cls._ordered_models(df['Model'])
        df['Model'] = pd.Categorical(df['Model'], categories=models)

        fig, ax = plt.subplots()
        sns.barplot(
            data=df,
            x='Group',
            y='Error',
            hue='Model',
            palette=cls._palette(models),
            width=0.9,
            errorbar=None,
            ax=ax,
        )
        ax.axhline(0.5, linestyle='--', color='red', linewidth=1.1)
        ax.set_xlabel('')
        ax.set_ylabel('Error')
        ax.tick_params(axis='x', labelsize=12)
        fig.tight_layout()
        return fig

    @classmethod
    def average_error_by_horizons(cls, df: pd.DataFrame) -> Figure:
        return cls._faceted_model_bars(df, y='Error', facet='Horizon', ylabel='SMAPE')

    @classmethod
    def average_error_by_stationarity(cls, df: pd.DataFrame, colname: str) -> Figure:
        df = df.rename(columns={'variable': 'Model'})
        return cls._faceted_model_bars(df, y='value', facet=colname, ylabel='SMAPE')

    @staticmethod
    def average_error_by_horizon_freq(df: pd.DataFrame) -> Figure:
        Plots.get_theme()
        g = sns.relplot(
            data=df,
            x='Horizon',
            y='Error',
            hue='Model',
            col='Frequency',
            kind='line',
            col_wrap=1,
            facet_kws={'sharex': False, 'sharey': False},
            linewidth=1,
        )
        g.tight_layout()
        return g.fig

    @classmethod
    def error_dist_by_model(cls, df: pd.DataFrame) -> Figure:
        Plots.get_theme()
        df_melted = df.sort_values('Error', ascending=False).reset_index(drop=True)
        models = cls._ordered_models(df_melted['Model'])
        df_melted['Model'] = pd.Categorical(df_melted['Model'], categories=models)

        fig, ax = plt.subplots()
        sns.boxplot(
            data=df_melted,
            x='Error',
            y='Model',
            hue='Model',
            palette=cls._palette(models),
            width=0.7,
            order=models,
            legend=False,
            ax=ax,
        )
        ax.set_xlabel('Error')
        ax.set_ylabel('Error distribution')
        ax.invert_yaxis()
        fig.tight_layout()
        return fig

    @staticmethod
    def result_with_rope_bars(df: pd.DataFrame) -> Figure:
        sns.set_theme(
            style='ticks',
            rc={
                'font.family': 'serif',
                'font.serif': ['Palatino', 'Palatino Linotype', 'Times New Roman', 'DejaVu Serif'],
                'axes.labelsize': 12,
                'xtick.labelsize': 10,
                'ytick.labelsize': 12,
            },
        )
        pivot = df.pivot_table(
            index='Model',
            columns='Result',
            values='Probability',
            aggfunc='sum',
        )
        n_results = pivot.shape[1]
        colors = sns.color_palette('husl', n_colors=max(n_results, 1))

        fig, ax = plt.subplots()
        pivot.plot(kind='barh', stacked=True, color=colors, width=0.8, ax=ax, legend=True)
        ax.set_xlabel('Proportion of probability')
        ax.set_ylabel('')
        ax.legend(title='', loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=max(n_results, 1))
        sns.despine(ax=ax)
        fig.tight_layout()
        return fig
