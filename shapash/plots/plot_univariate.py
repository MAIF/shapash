import warnings
from typing import Optional

import numpy as np
import pandas as pd
from plotly import graph_objs as go
from plotly.offline import plot
from scipy.stats import gaussian_kde

from shapash.report.common import VarType, series_dtype
from shapash.style.style_utils import define_style, get_palette, random_color
from shapash.utils.utils import adjust_title_height, compute_digit_number


def plot_distribution(
    df_all: pd.DataFrame,
    col: str,
    hue: Optional[str] = None,
    colors_dict: Optional[dict] = None,
    width: int = 700,
    height: int = 500,
    palette_name: str = "default",
    nb_cat_max: int = 7,
    nb_hue_max: int = 7,
    file_name=None,
    auto_open=False,
) -> go.Figure:
    """
    Generate a Plotly figure displaying the univariate distribution of a feature
    (continuous or categorical) in the dataset.

    For categorical features with too many unique categories, the least frequent
    categories are grouped into a new 'Other' category to ensure the plot remains
    readable. Continuous features are visualized using KDE plots.

    The input DataFrame must contain the column of interest (`col`) and a second column
    (`hue`) used to distinguish between two groups (e.g., 'train' and 'test').

    Parameters
    ----------
    df_all : pd.DataFrame
        The input DataFrame containing the data to be plotted.
    col : str
        The name of the column of interest whose distribution is to be visualized.
    hue : Optional[str], optional
        The name of the column used to differentiate between groups (e.g., 'train' and 'test').
    colors_dict : Optional[dict], optional
        A dictionary specifying the colors to be used for each group. If not provided,
        a default color palette will be used.
    width : int, optional, default=700
        The width of the generated figure, in pixels.
    height : int, optional, default=500
        The height of the generated figure, in pixels.
    palette_name : str, optional, default="default"
        The name of the color palette to be used if `colors_dict` is not provided.
    nb_cat_max : int, optional, default=7
        Maximum number of categories to display. Categories beyond this limit
        are grouped into a new 'Other' category (only for categorical features).
    nb_hue_max : int, optional, default=7
        Maximum number of hue categories to display. Categories beyond this limit
        are grouped into a new 'Other' category.
    file_name : str, optional
        Path to save the plot as an HTML file. If None, the plot will not be saved, by default None.
    auto_open : bool, optional
        If True, the plot will automatically open in a web browser after being generated, by default False.

    Returns
    -------
    go.Figure
        A Plotly figure object representing the distribution of the feature.
    """

    serie_type = series_dtype(df_all[col])

    if col not in df_all.columns:
        raise ValueError(f"Column '{col}' not found in the input DataFrame.")

    if hue is not None and hue not in df_all.columns:
        raise ValueError(f"Column '{hue}' not found in the input DataFrame.")

    if serie_type == VarType.TYPE_NUM:
        # Use the continuous plotting function
        fig = plot_continuous_distribution(
            df_all,
            col,
            hue=hue,
            colors_dict=colors_dict,
            width=width,
            height=height,
            palette_name=palette_name,
            nb_hue_max=nb_hue_max,
            file_name=file_name,
            auto_open=auto_open,
        )
    elif serie_type == VarType.TYPE_CAT:
        # Use the categorical plotting function
        fig = plot_categorical_distribution(
            df_all,
            col,
            hue=hue,
            colors_dict=colors_dict,
            width=width,
            height=height,
            palette_name=palette_name,
            nb_cat_max=nb_cat_max,
            nb_hue_max=nb_hue_max,
            file_name=file_name,
            auto_open=auto_open,
        )
    else:
        raise NotImplementedError("The specified series type is not supported.")
    return fig


def plot_continuous_distribution(
    df_all: pd.DataFrame,
    col: str,
    hue: Optional[str] = None,
    colors_dict: Optional[dict] = None,
    width: int = 700,
    height: int = 500,
    palette_name: str = "default",
    nb_hue_max: int = 7,
    file_name=None,
    auto_open=False,
) -> go.Figure:
    """
    Returns a Plotly figure containing the distribution of a continuous feature.

    Parameters
    ----------
    df_all : pd.DataFrame
        The input dataframe that contains the column of interest
    col : str
        The column of interest
    hue : Optional[str]
        The column used to distinguish the values (e.g., 'train' and 'test').
    colors_dict : Optional[dict]
        Dictionary of colors for hue levels.
    width : int, optional, default=700
        The width of the generated figure, in pixels.
    height : int, optional, default=500
        The height of the generated figure, in pixels.
    palette_name : str, optional, default="default"
        The name of the color palette to use if `colors_dict` is not provided.
    nb_hue_max : int, optional, default=7
        Maximum number of hue categories to display. Categories beyond this limit
        are grouped into a new 'Other' category.
    file_name : str, optional
        Path to save the plot as an HTML file. If None, the plot will not be saved, by default None.
    auto_open : bool, optional
        If True, the plot will automatically open in a web browser after being generated, by default False.

    Returns
    -------
    go.Figure
        Plotly figure object representing the KDE plot.
    """
    if colors_dict:
        style_dict = {}
        keys = ["dict_title", "init_confusion_matrix_colorscale", "dict_xaxis", "dict_yaxis"]
        if any(key not in colors_dict for key in keys):
            style_dict = define_style(get_palette(palette_name))
        style_dict.update(colors_dict)
    else:
        style_dict = define_style(get_palette(palette_name))

    filtered_data = df_all.copy()
    if len(filtered_data) > 200:
        lower_quantile = filtered_data[col].quantile(0.005)
        upper_quantile = filtered_data[col].quantile(0.995)
        filtered_data = filtered_data[(filtered_data[col] > lower_quantile) & (filtered_data[col] < upper_quantile)]

    # Initialize the figure
    fig = go.Figure()

    # Define colors for hue levels if provided
    if hue:
        unique_hues = filtered_data[hue].unique()

        if len(unique_hues) > nb_hue_max:
            top_categories = filtered_data[hue].value_counts().nlargest(nb_hue_max).index
            filtered_data[hue] = filtered_data[hue].where(filtered_data[hue].isin(top_categories), other="Other")
            unique_hues = filtered_data[hue].unique()

        for level in unique_hues:
            subset = filtered_data[filtered_data[hue] == level]
            if len(subset) < 5:
                warnings.warn(  # noqa: B028
                    f"Not enough data points to plot the curve for level '{level}' in the hue column '{hue}'. "
                    "At least 5 data points are required."
                )
                continue
            kde = gaussian_kde(subset[col])
            x_values = np.linspace(subset[col].min(), subset[col].max(), 500)
            y_values = kde(x_values)

            # Generate hovertext
            hv_text = [
                f"{hue}: {level}<br>"
                f"{col}: {format(x, f'.{max(0, compute_digit_number(x, 3))}f')}<br>"
                f"Density: {y:.4f}"
                for x, y in zip(x_values, y_values)
            ]

            color = style_dict.get(level, random_color())

            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=y_values,
                    mode="lines",
                    line=dict(color=color),
                    name=str(level),
                    fill="tozeroy",
                    hoverinfo="text",
                    text=hv_text,
                )
            )
    else:
        if len(filtered_data[col]) < 5:
            warnings.warn(  # noqa: B028
                f"Not enough data points to plot the curve in the hue column '{hue}'. "
                "At least 5 data points are required."
            )
            return
        kde = gaussian_kde(filtered_data[col])
        x_values = np.linspace(filtered_data[col].min(), filtered_data[col].max(), 500)
        y_values = kde(x_values)

        # Generate hovertext
        hv_text = [
            f"{col}: {format(x, f'.{max(0, compute_digit_number(x, 3))}f')}<br>" f"Density: {y:.4f}"
            for x, y in zip(x_values, y_values)
        ]

        color = style_dict.get(col, random_color())

        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="lines",
                line=dict(color=color),
                name=col,
                fill="tozeroy",
                hoverinfo="text",
                text=hv_text,
            )
        )

    title = f"Distribution of {col}"
    dict_t = style_dict["dict_title"] | dict(text=title, y=adjust_title_height(height))
    dict_xaxis = style_dict["dict_xaxis"] | dict(text=col)
    dict_yaxis = style_dict["dict_yaxis"] | dict(text="Density")

    # Update layout
    fig.update_layout(
        title=dict_t,
        xaxis=dict(title=dict_xaxis, tickangle=30),
        yaxis=dict(title=dict_yaxis),
        width=width,
        height=height,
        margin=dict(l=90, r=20, t=100, b=70),
        template="plotly_white",
    )

    if hue:
        fig.update_layout(
            legend_title=dict(
                text=hue,
                font=dict(size=12),
            )
        )

    if file_name:
        plot(fig, filename=file_name, auto_open=auto_open)

    return fig


def plot_categorical_distribution(
    df_all: pd.DataFrame,
    col: str,
    hue: Optional[str] = None,
    nb_cat_max: int = 7,
    nb_hue_max: int = 7,
    colors_dict: Optional[dict] = None,
    width: int = 700,
    height: int = 500,
    palette_name: str = "default",
    file_name=None,
    auto_open=False,
) -> go.Figure:
    """
    Returns a Plotly Figure containing the distribution of a categorical feature.

    If the feature contains too many categories, the smallest categories are grouped
    into a new 'Other' category so that the graph remains readable.

    Parameters
    ----------
    df_all : pd.DataFrame
        The input dataframe that contains the column of interest.
    col : str
        The column of interest.
    hue : Optional[str]
        The column used to distinguish the values (e.g., 'train' and 'test').
    nb_cat_max : int, optional, default=7
        Maximum number of categories to display. Categories beyond this limit
        are grouped into a new 'Other' category.
    nb_hue_max : int, optional, default=7
        Maximum number of hue categories to display. Categories beyond this limit
        are grouped into a new 'Other' category.
    colors_dict : Optional[dict]
        Dictionary of colors for categories.
    width : int, optional, default=700
        The width of the generated figure, in pixels.
    height : int, optional, default=500
        The height of the generated figure, in pixels.
    palette_name : str, optional, default="default"
        The name of the color palette to use if `colors_dict` is not provided.
    file_name : str, optional
        Path to save the plot as an HTML file. If None, the plot will not be saved, by default None.
    auto_open : bool, optional
        If True, the plot will automatically open in a web browser after being generated, by default False.

    Returns
    -------
    go.Figure
        Plotly figure object representing the bar plot.
    """
    df_all = df_all.copy()
    if colors_dict:
        style_dict = {}
        keys = ["dict_title", "init_confusion_matrix_colorscale", "dict_xaxis", "dict_yaxis"]
        if any(key not in colors_dict for key in keys):
            style_dict = define_style(get_palette(palette_name))
        style_dict.update(colors_dict)
    else:
        style_dict = define_style(get_palette(palette_name))

    if hue:
        unique_hues = df_all[hue].unique()
        if len(unique_hues) > nb_hue_max:
            top_categories = df_all[hue].value_counts().nlargest(nb_hue_max).index
            df_all[hue] = df_all[hue].where(df_all[hue].isin(top_categories), other="Other")

        df_cat = df_all.groupby([col, hue])[col].count().rename("count").reset_index()
        df_cat["Percent"] = df_cat["count"] * 100 / df_cat.groupby(hue)["count"].transform("sum")
    else:
        df_cat = df_all[col].value_counts().reset_index(name="count")
        df_cat["Percent"] = df_cat["count"] * 100 / df_cat["count"].sum()

    if pd.api.types.is_numeric_dtype(df_cat[col].dtype):
        df_cat = df_cat.sort_values(col, ascending=True)
        df_cat[col] = df_cat[col].astype(str)

    nb_cat = df_cat[col].nunique()

    if nb_cat > nb_cat_max:
        df_cat = _merge_small_categories(df_cat=df_cat, col=col, hue=hue, nb_cat_max=nb_cat_max)

    total_counts = df_cat.groupby(col)["count"].sum()
    category_order = total_counts.sort_values().index
    if hue:
        hue_order = np.sort(df_cat[hue].unique())
        full_combinations = pd.MultiIndex.from_product([category_order, hue_order], names=[col, hue])
        df_cat = df_cat.set_index([col, hue]).reindex(full_combinations, fill_value=0).reset_index()
    else:
        df_cat = df_cat.set_index(col).reindex(category_order, fill_value=0).reset_index()

    df_cat[col] = pd.Categorical(df_cat[col], categories=category_order, ordered=True)

    data = []
    if hue:
        for hue_val in hue_order:
            subset = df_cat[df_cat[hue] == hue_val]
            color = style_dict.get(hue_val, random_color())

            customdata = subset.apply(
                lambda row, hue_val=hue_val: (
                    f"{hue}: {hue_val}<br>"
                    f"{col}: {row[col]}<br>"
                    f"Percentage: {format(row.Percent, f'.{max(0, compute_digit_number(row.Percent, 3))}f')}%"
                ),
                axis=1,
            )

            bar = go.Bar(
                x=subset["Percent"],
                y=subset[col],
                orientation="h",
                name=str(hue_val),
                marker=dict(color=color),
                customdata=customdata,
                hovertemplate="%{customdata}<extra></extra>",
            )
            data.append(bar)
    else:
        color = style_dict.get(col, random_color())

        customdata = subset.apply(
            lambda row: (
                f"{col}: {row[col]}<br>"
                f"Percentage: {format(row.Percent, f'.{max(0, compute_digit_number(row.Percent, 3))}f')}%"
            ),
            axis=1,
        )

        bar = go.Bar(
            x=df_cat["Percent"],
            y=df_cat[col],
            orientation="h",
            name=col,
            marker=dict(color=color),
            customdata=customdata,
            hovertemplate="%{customdata}<extra></extra>",
        )
        data = [bar]

    fig = go.Figure(data=data)

    title = f"Distribution of {col}"
    dict_t = style_dict["dict_title"] | dict(text=title, y=adjust_title_height(height))
    dict_xaxis = style_dict["dict_xaxis"] | dict(text="Percentage")
    dict_yaxis = style_dict["dict_yaxis"] | dict(text=col)

    fig.update_layout(
        title=dict_t,
        xaxis_title=dict_xaxis,
        yaxis_title=dict_yaxis,
        barmode="group",
        width=width,
        height=height,
        margin=dict(l=110, r=20, t=100, b=70),
        template="plotly_white",
    )

    # Add legend title only if hue is specified
    if hue:
        fig.update_layout(
            legend_title=dict(
                text=hue,
                font=dict(size=12),
            )
        )

    if file_name:
        plot(fig, filename=file_name, auto_open=auto_open)

    return fig


def _merge_small_categories(df_cat: pd.DataFrame, col: str, hue: Optional[str], nb_cat_max: int) -> pd.DataFrame:
    """
    Merges smaller categories into a single "Other" category.

    Parameters
    ----------
    df_cat : pd.DataFrame
        Dataframe with category counts and percentages.
    col : str
        The column of interest.
    hue : Optional[str]
        Hue column to group by.
    nb_cat_max : int
        Maximum number of categories to retain.

    Returns
    -------
    pd.DataFrame
        Dataframe with smaller categories grouped into "Other".
    """
    total_counts = df_cat.groupby(col)["count"].sum()
    sorted_categories = total_counts.sort_values(ascending=False).index
    top_categories = sorted_categories[:nb_cat_max]

    df_cat[col] = np.where(df_cat[col].isin(top_categories), df_cat[col], "Other")
    if hue:
        df_cat = df_cat.groupby([col, hue]).agg({"count": "sum"}).reset_index()
        df_cat["Percent"] = df_cat["count"] * 100 / df_cat.groupby(hue)["count"].transform("sum")
    else:
        df_cat = df_cat.groupby(col, as_index=False)["count"].sum()
        df_cat["Percent"] = df_cat["count"] * 100 / df_cat["count"].sum()

    return df_cat
