from typing import Union
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from shapash.utils.utils import truncate_str
from shapash.report.common import VarType

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

cmap_diverging = LinearSegmentedColormap.from_list('col_corr', col_scale, N=100)

cmap_gradient = LinearSegmentedColormap.from_list('col_corr', col_scale[4:6], N=100)

dict_color_palette = {'train': (74/255, 99/255, 138/255, 0.7), 'test': (244/255, 192/255, 0),
                      'true': (74/255, 99/255, 138/255, 0.7), 'pred': (244/255, 192/255, 0)}


def generate_fig_univariate(df_all: pd.DataFrame, col: str, hue: str, type: VarType) -> plt.Figure:
    """
    Returns a matplotlib figure containing the distribution of any kind of feature
    (continuous, categorical).

    If the feature is categorical and contains too many categories, the smallest
    categories are grouped into a new 'Other' category so that the graph remains
    readable.

    The input dataframe should contain the column of interest and a column that is used
    to distinguish two types of values (ex. 'train' and 'test')

    Parameters
    ----------
    df_all : pd.DataFrame
        The input dataframe that contains the column of interest
    col : str
        The column of interest
    hue : str
        The column used to distinguish the values (ex. 'train' and 'test')
    type: str
        The type of the series ('continous' or 'categorical')

    Returns
    -------
    matplotlib.pyplot.Figure
    """
    if type == VarType.TYPE_NUM:
        fig = generate_fig_univariate_continuous(df_all, col, hue=hue)
    elif type == VarType.TYPE_CAT:
        fig = generate_fig_univariate_categorical(df_all, col, hue=hue)
    else:
        raise NotImplemented("Series dtype not supported")
    return fig


def generate_fig_univariate_continuous(df_all: pd.DataFrame, col: str, hue: str) -> plt.Figure:
    """
    Returns a matplotlib figure containing the distribution of a continuous feature.

    Parameters
    ----------
    df_all : pd.DataFrame
        The input dataframe that contains the column of interest
    col : str
        The column of interest
    hue : str
        The column used to distinguish the values (ex. 'train' and 'test')

    Returns
    -------
    matplotlib.pyplot.Figure
    """
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
    """
    Returns a matplotlib figure containing the distribution of a categorical feature.

    If the feature is categorical and contains too many categories, the smallest
    categories are grouped into a new 'Other' category so that the graph remains
    readable.

    Parameters
    ----------
    df_all : pd.DataFrame
        The input dataframe that contains the column of interest
    col : str
        The column of interest
    hue : str
        The column used to distinguish the values (ex. 'train' and 'test')
    nb_cat_max : int
        The number max of categories to be displayed. If the number of categories
        is greater than nb_cat_max then groups smallest categories into a new
        'Other' category

    Returns
    -------
    matplotlib.pyplot.Figure
    """
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

    new_labels = [truncate_str(i.get_text(), maxlen=45) for i in ax.yaxis.get_ticklabels()]
    ax.yaxis.set_ticklabels(new_labels)

    return fig


def _merge_small_categories(df_cat: pd.DataFrame, col: str, hue: str,  nb_cat_max: int) -> pd.DataFrame:
    """
    Merges categories of column 'col' of df_cat into 'Other' category so that
    the number of categories is less than nb_cat_max.
    """
    df_cat_sum_hue = df_cat.groupby([col]).agg({'count': 'sum'}).reset_index()
    list_cat_to_merge = df_cat_sum_hue.sort_values('count', ascending=False)[col].to_list()[nb_cat_max - 1:]
    df_cat_other = df_cat.loc[df_cat[col].isin(list_cat_to_merge)] \
        .groupby(hue, as_index=False)[["count", "Percent"]].sum()
    df_cat_other[col] = "Other"
    return df_cat.loc[~df_cat[col].isin(list_cat_to_merge)].append(df_cat_other)


def generate_confusion_matrix_plot(y_true: Union[np.array, list], y_pred: Union[np.array, list]) -> plt.Figure:
    """
    Returns a matplotlib figure containing a confusion matrix that is computed using y_true and
    y_pred parameters.

    Parameters
    ----------
    y_true : array-like
        Ground truth (correct) target values.
    y_pred : array-like
        Estimated targets as returned by a classifier.

    Returns
    -------
    matplotlib.pyplot.Figure
    """
    df_cm = pd.crosstab(y_true, y_pred, rownames=['Actual'], colnames=['Predicted'])
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.heatmap(df_cm, ax=ax, annot=True, cmap=cmap_gradient, fmt='g')
    return fig
