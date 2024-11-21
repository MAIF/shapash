from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from shapash.report.common import VarType
from shapash.style.style_utils import get_palette, get_pyplot_color
from shapash.utils.utils import truncate_str


def generate_fig_univariate(
    df_all: pd.DataFrame,
    col: str,
    hue: str,
    serie_type: VarType,
    colors_dict: Optional[dict] = None,
    width: int = 7,
    height: int = 4,
    palette_name: str = "default",
) -> plt.Figure:
    """
    Generate a matplotlib figure displaying the univariate distribution of a feature
    (continuous or categorical) in the dataset.

    For categorical features with too many unique categories, the least frequent
    categories are grouped into a new 'Other' category to ensure the plot remains
    readable. Continuous features are visualized using histograms.

    The input DataFrame must contain the column of interest (`col`) and a second column
    (`hue`) used to distinguish between two groups (e.g., 'train' and 'test').

    Parameters
    ----------
    df_all : pd.DataFrame
        The input DataFrame containing the data to be plotted.
    col : str
        The name of the column of interest whose distribution is to be visualized.
    hue : str
        The name of the column used to differentiate between groups (e.g., 'train' and 'test').
    serie_type : VarType
        The type of the feature, either 'continuous' or 'categorical'.
    colors_dict : dict, optional
        A dictionary specifying the colors to be used for each group. If not provided,
        a default color palette will be used.
    width : int, optional, default=7
        The width of the generated figure, in inches.
    height : int, optional, default=4
        The height of the generated figure, in inches.
    palette_name : str, optional, default="default"
        The name of the color palette to be used if `colors_dict` is not provided.

    Returns
    -------
    matplotlib.pyplot.Figure
        A matplotlib figure object representing the distribution of the feature.
    """
    if serie_type == VarType.TYPE_NUM:
        fig = generate_fig_univariate_continuous(
            df_all, col, hue=hue, colors_dict=colors_dict, width=width, height=height, palette_name=palette_name
        )
    elif serie_type == VarType.TYPE_CAT:
        fig = generate_fig_univariate_categorical(
            df_all, col, hue=hue, colors_dict=colors_dict, width=width, height=height, palette_name=palette_name
        )
    else:
        raise NotImplementedError("Series dtype not supported")
    return fig


def generate_fig_univariate_continuous(
    df_all: pd.DataFrame,
    col: str,
    hue: str,
    colors_dict: Optional[dict] = None,
    width: int = 7,
    height: int = 4,
    palette_name: str = "default",
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
    width : int, optional, default=7
        The width of the generated figure, in inches.
    height : int, optional, default=4
        The height of the generated figure, in inches.
    palette_name : str, optional, default="default"
        The name of the color palette to be used if `colors_dict` is not provided.

    Returns
    -------
    matplotlib.pyplot.Figure
    """
    colors_dict = colors_dict or get_palette(palette_name)
    lower_quantile = df_all[:, col].quantile(0.005)
    upper_quantile = df_all[:, col].quantile(0.995)
    cond = (df_all[col] > lower_quantile) & (df_all[col] < upper_quantile)

    g = sns.displot(
        df_all[cond],
        x=col,
        hue=hue,
        kind="kde",
        fill=True,
        common_norm=False,
        palette=get_pyplot_color(colors=colors_dict["report_feature_distribution"]),
    )
    g.set_xticklabels(rotation=30)

    fig = g.figure

    fig.set_figwidth(width)
    fig.set_figheight(height)

    return fig


def generate_fig_univariate_categorical(
    df_all: pd.DataFrame,
    col: str,
    hue: str,
    nb_cat_max: int = 7,
    colors_dict: Optional[dict] = None,
    width: int = 7,
    height: int = 4,
    palette_name: str = "default",
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
    width : int, optional, default=7
        The width of the generated figure, in inches.
    height : int, optional, default=4
        The height of the generated figure, in inches.
    palette_name : str, optional, default="default"
        The name of the color palette to be used if `colors_dict` is not provided.

    Returns
    -------
    matplotlib.pyplot.Figure
    """
    colors_dict = colors_dict or get_palette(palette_name)
    df_cat = df_all.groupby([col, hue]).agg({col: "count"}).rename(columns={col: "count"}).reset_index()
    df_cat["Percent"] = df_cat["count"] * 100 / df_cat.groupby(hue)["count"].transform("sum")

    if pd.api.types.is_numeric_dtype(df_cat[col].dtype):
        df_cat = df_cat.sort_values(col, ascending=True)
        df_cat[col] = df_cat[col].astype(str)

    nb_cat = df_cat.groupby([col]).agg({"count": "sum"}).reset_index()[col].nunique()

    if nb_cat > nb_cat_max:
        df_cat = _merge_small_categories(df_cat=df_cat, col=col, hue=hue, nb_cat_max=nb_cat_max)

    fig, ax = plt.subplots(figsize=(width, height))

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
            f"{np.nan_to_num(p.get_width(), nan=0):.1f}%",
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
