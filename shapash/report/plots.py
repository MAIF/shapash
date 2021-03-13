import pandas as pd
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
