import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from shapash.utils.utils import truncate_str
from shapash.report.common import numeric_is_continuous, series_dtype, VarType

# Color scale derivated from SmartPlotter init_color_scale attribute
col_scale = [(0.204, 0.216, 0.212),
             (0.29, 0.388, 0.541),
             (0.455, 0.6, 0.839),
             (0.635, 0.737, 0.835),
             (1, 1, 1),
             (0.957, 0.753, 0.0),
             (1.0, 0.651, 0.067),
             (1.0, 0.482, 0.149),
             (1.0, 0.302, 0.027)]

dict_color_palette = {'train': (74/255, 99/255, 138/255, 0.7), 'test': (244/255, 192/255, 0),
                      'true': (74/255, 99/255, 138/255, 0.7), 'pred': (244/255, 192/255, 0)}


def generate_fig_univariate(df_all: pd.DataFrame, col: str, hue: str) -> plt.Figure:
    s_dtype = series_dtype(df_all[col])
    if s_dtype == VarType.TYPE_NUM:
        if numeric_is_continuous(df_all[col]):
            fig = generate_fig_univariate_continuous(df_all, col, hue=hue)
        else:
            fig = generate_fig_univariate_categorical(df_all, col, hue=hue)
    elif s_dtype == VarType.TYPE_CAT:
        fig = generate_fig_univariate_categorical(df_all, col, hue=hue)
    else:
        fig = plt.Figure()

    return fig


def generate_fig_univariate_continuous(df_all: pd.DataFrame, col: str, hue: str) -> plt.Figure:
    g = sns.displot(df_all, x=col, hue=hue, kind="kde", fill=True, common_norm=False,
                    palette=dict_color_palette)
    g.set_xticklabels(rotation=30)

    fig = g.fig

    fig.set_figwidth(7)
    fig.set_figheight(4)

    return fig


def generate_fig_univariate_categorical(
        df_all: pd.DataFrame,
        col: str,
        hue: str,
        nb_cat_max: int = 7,
) -> plt.Figure:
    df_cat = df_all.groupby([col, hue]).agg({col: 'count'})\
                   .rename(columns={col: "count"}).reset_index()
    df_cat['Percent'] = df_cat['count'] * 100 / df_cat.groupby(hue)['count'].transform('sum')

    if pd.api.types.is_numeric_dtype(df_cat[col].dtype):
        df_cat = df_cat.sort_values(col, ascending=True)
        df_cat[col] = df_cat[col].astype(str)

    nb_cat = df_cat.groupby([col]).agg({'count': 'sum'}).reset_index()[col].nunique()

    if nb_cat > nb_cat_max:
        df_cat = _merge_small_categories(df_cat=df_cat, col=col, hue=hue, nb_cat_max=nb_cat_max)

    fig, ax = plt.subplots(figsize=(7, 4))

    sns.barplot(data=df_cat, x='Percent', y=col, hue=hue,
                palette=dict_color_palette, ax=ax)

    for p in ax.patches:
        ax.annotate("{:.1f}%".format(np.nan_to_num(p.get_width(), nan=0)),
                    xy=(p.get_width(), p.get_y() + p.get_height() / 2),
                    xytext=(5, 0), textcoords='offset points', ha="left", va="center")

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Removes plot borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    new_labels = [truncate_str(i.get_text(), maxlen=15) for i in ax.yaxis.get_ticklabels()]
    ax.yaxis.set_ticklabels(new_labels)

    return fig


def _merge_small_categories(df_cat: pd.DataFrame, col: str, hue: str,  nb_cat_max: int) -> pd.DataFrame:
    df_cat_sum_hue = df_cat.groupby([col]).agg({'count': 'sum'}).reset_index()
    list_cat_to_merge = df_cat_sum_hue.sort_values('count', ascending=False)[col].to_list()[nb_cat_max - 1:]
    df_cat_other = df_cat.loc[df_cat[col].isin(list_cat_to_merge)] \
        .groupby(hue, as_index=False)[["count", "Percent"]].sum()
    df_cat_other[col] = "Other"
    return df_cat.loc[~df_cat[col].isin(list_cat_to_merge)].append(df_cat_other)


def generate_correlation_matrix_fig(df_train_test: pd.DataFrame):

    def generate_unique_corr_fig(df: pd.DataFrame, ax: plt.Axes):
        sns.set_theme(style="white")
        corr = df.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        cmap = LinearSegmentedColormap.from_list('col_corr',
                                                 col_scale,
                                                 N=100)
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
    fig = g.fig
    fig.suptitle('Scatter plot matrix', fontsize=20, x=0.45)
    plt.tight_layout()

    return fig
