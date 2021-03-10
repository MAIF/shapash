import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from shapash.report.common import numeric_is_continuous, series_dtype, VarType


def generate_fig_univariate(df_train_test: pd.DataFrame, col: str) -> plt.Figure:
    df_train_test = df_train_test.copy()
    s_dtype = series_dtype(df_train_test[col])
    if s_dtype == VarType.TYPE_NUM:
        if numeric_is_continuous(df_train_test[col]):
            fig = generate_fig_univariate_continuous(df_train_test, col)
        else:
            fig = generate_fig_univariate_categorical(df_train_test, col)
    elif s_dtype == VarType.TYPE_CAT:
        df_train_test.loc[:, col] = df_train_test.loc[:, col].apply(lambda x: x[:13] + '..' if len(x) > 15 else x)
        fig = generate_fig_univariate_categorical(df_train_test, col)
    else:
        fig = plt.Figure()

    fig.set_figwidth(5)
    fig.set_figheight(4)

    return fig


def generate_fig_univariate_continuous(df_train_test: pd.DataFrame, col: str) -> plt.Figure:
    g = sns.displot(df_train_test, x=col, hue="data_train_test", kind="kde", fill=True, common_norm=False)
    g.set_xticklabels(rotation=30)

    return g.fig


def generate_fig_univariate_categorical(df_train_test: pd.DataFrame, col: str) -> plt.Figure:
    g = sns.displot(data=df_train_test, x=col, hue='data_train_test', kind='hist', stat='probability',
                    common_norm=False, multiple="dodge", shrink=.8, alpha=0.3)
    g.set_xticklabels(rotation=30)
    return g.fig


def generate_correlation_matrix_fig(df_train_test: pd.DataFrame):

    def generate_unique_corr_fig(df: pd.DataFrame, ax: plt.Axes):
        sns.set_theme(style="white")
        corr = df.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        sns.heatmap(corr, mask=mask, cmap=cmap, center=0, ax=ax,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5})

    if df_train_test['data_train_test'].nunique() > 1:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))
        generate_unique_corr_fig(
            df=df_train_test.loc[df_train_test.data_train_test == 'train'].drop('data_train_test', axis=1),
            ax=ax1
        )
        generate_unique_corr_fig(
            df=df_train_test.loc[df_train_test.data_train_test == 'test'].drop('data_train_test', axis=1),
            ax=ax2
        )
        ax1.set_title("Train")
        ax2.set_title("Test")
    else:
        fig, ax = plt.subplots(figsize=(11, 9))
        generate_unique_corr_fig(
            df=df_train_test.drop('data_train_test', axis=1),
            ax=ax
        )
        ax.set_title("Test")
    fig.suptitle('Correlation matrix', fontsize=20, x=0.45)
    return fig


def generate_scatter_plot_fig(df_train_test: pd.DataFrame):
    sns.set_theme(style="ticks")

    g = sns.pairplot(df_train_test, hue="data_train_test")

    return g.fig
