from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from shapash.report.common import VarType
from shapash.style.style_utils import get_palette, get_pyplot_color
from shapash.utils.utils import truncate_str


def generate_fig_univariate(
    df_all: pd.DataFrame, col: str, hue: str, type: VarType, colors_dict: Optional[dict] = None
) -> plt.Figure:
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
    colors_dict : dict
        dict of colors used

    Returns
    -------
    matplotlib.pyplot.Figure
    """
    if type == VarType.TYPE_NUM:
        fig = generate_fig_univariate_continuous(df_all, col, hue=hue, colors_dict=colors_dict)
    elif type == VarType.TYPE_CAT:
        fig = generate_fig_univariate_categorical(df_all, col, hue=hue, colors_dict=colors_dict)
    else:
        raise NotImplementedError("Series dtype not supported")
    return fig


def generate_fig_univariate_continuous(
    df_all: pd.DataFrame, col: str, hue: str, colors_dict: Optional[dict] = None
) -> plt.Figure:
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
    colors_dict : dict
        dict of colors used

    Returns
    -------
    matplotlib.pyplot.Figure
    """
    colors_dict = colors_dict or get_palette("default")
    g = sns.displot(
        df_all,
        x=col,
        hue=hue,
        kind="kde",
        fill=True,
        common_norm=False,
        palette=get_pyplot_color(colors=colors_dict["report_feature_distribution"]),
    )
    g.set_xticklabels(rotation=30)

    fig = g.fig

    fig.set_figwidth(7)
    fig.set_figheight(4)

    return fig


def generate_fig_univariate_categorical(
    df_all: pd.DataFrame, col: str, hue: str, nb_cat_max: int = 7, colors_dict: Optional[dict] = None
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
    colors_dict : dict
        dict of colors used

    Returns
    -------
    matplotlib.pyplot.Figure
    """
    colors_dict = colors_dict or get_palette("default")
    df_cat = df_all.groupby([col, hue]).agg({col: "count"}).rename(columns={col: "count"}).reset_index()
    df_cat["Percent"] = df_cat["count"] * 100 / df_cat.groupby(hue)["count"].transform("sum")

    if pd.api.types.is_numeric_dtype(df_cat[col].dtype):
        df_cat = df_cat.sort_values(col, ascending=True)
        df_cat[col] = df_cat[col].astype(str)

    nb_cat = df_cat.groupby([col]).agg({"count": "sum"}).reset_index()[col].nunique()

    if nb_cat > nb_cat_max:
        df_cat = _merge_small_categories(df_cat=df_cat, col=col, hue=hue, nb_cat_max=nb_cat_max)

    fig, ax = plt.subplots(figsize=(7, 4))

    sns.barplot(
        data=df_cat,
        x="Percent",
        y=col,
        hue=hue,
        palette=get_pyplot_color(colors=colors_dict["report_feature_distribution"]),
        ax=ax,
    )

    for p in ax.patches:
        ax.annotate(
            "{:.1f}%".format(np.nan_to_num(p.get_width(), nan=0)),
            xy=(p.get_width(), p.get_y() + p.get_height() / 2),
            xytext=(5, 0),
            textcoords="offset points",
            ha="left",
            va="center",
        )

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    # Removes plot borders
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    new_labels = [truncate_str(i.get_text(), maxlen=45) for i in ax.yaxis.get_ticklabels()]
    ax.yaxis.set_ticklabels(new_labels)

    return fig


def _merge_small_categories(df_cat: pd.DataFrame, col: str, hue: str, nb_cat_max: int) -> pd.DataFrame:
    """
    Merges categories of column 'col' of df_cat into 'Other' category so that
    the number of categories is less than nb_cat_max.
    """
    df_cat_sum_hue = df_cat.groupby([col]).agg({"count": "sum"}).reset_index()
    list_cat_to_merge = df_cat_sum_hue.sort_values("count", ascending=False)[col].to_list()[nb_cat_max - 1 :]
    df_cat_other = (
        df_cat.loc[df_cat[col].isin(list_cat_to_merge)].groupby(hue, as_index=False)[["count", "Percent"]].sum()
    )
    df_cat_other[col] = "Other"
    return pd.concat([df_cat.loc[~df_cat[col].isin(list_cat_to_merge)], df_cat_other], axis=0)


def generate_confusion_matrix_plot(
    y_true: Union[np.array, list], y_pred: Union[np.array, list], colors_dict: Optional[dict] = None
) -> plt.Figure:
    """
    Returns a matplotlib figure containing a confusion matrix that is computed using y_true and
    y_pred parameters.

    Parameters
    ----------
    y_true : array-like
        Ground truth (correct) target values.
    y_pred : array-like
        Estimated targets as returned by a classifier.
    colors_dict : dict
        dict of colors used
    Returns
    -------
    matplotlib.pyplot.Figure
    """
    colors_dict = colors_dict or get_palette("default")
    col_scale = get_pyplot_color(colors=colors_dict["report_confusion_matrix"])
    cmap_gradient = LinearSegmentedColormap.from_list("col_corr", col_scale, N=100)

    df_cm = pd.crosstab(y_true, y_pred, rownames=["Actual"], colnames=["Predicted"])
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.heatmap(df_cm, ax=ax, annot=True, cmap=cmap_gradient, fmt="g")
    return fig
