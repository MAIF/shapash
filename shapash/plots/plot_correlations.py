from collections.abc import Iterable
from typing import Any, Literal

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from plotly import graph_objs as go
from plotly.offline import plot
from plotly.subplots import make_subplots
from scipy.spatial.distance import pdist

from shapash.manipulation.summarize import compute_corr, contribution_weighted_corr_matrix
from shapash.style.style_utils import define_style, get_palette
from shapash.utils.dtypes import text_like_columns
from shapash.utils.utils import adjust_title_height, compute_top_correlations_features, suffix_duplicates


def _cluster_corr(
    corr: pd.DataFrame | np.ndarray,
    degree: float,
    inplace: bool = False,
) -> pd.DataFrame | np.ndarray:
    """
    Rearrange a correlation matrix so that highly correlated variables are
    grouped together.

    Parameters
    ----------
    corr : pd.DataFrame | np.ndarray
        NxN correlation matrix.
    degree : float
        Exponent applied to the absolute correlation values to emphasize
        stronger correlations during clustering.
    inplace : bool, default=False
        Whether to modify the provided matrix in place.

    Returns
    -------
    pd.DataFrame | np.ndarray
        Correlation matrix with reordered rows and columns.
    """
    if corr.shape[0] < 2:
        return corr

    pairwise_distances = pdist(np.abs(corr) ** degree)

    finite_mask = np.isfinite(pairwise_distances)
    if not np.all(finite_mask):
        max_valid = pairwise_distances[finite_mask].max() if np.any(finite_mask) else 0.0
        pairwise_distances[~finite_mask] = max_valid

    linkage = sch.linkage(pairwise_distances, method="complete")
    cluster_distance_threshold = pairwise_distances.max() / 2
    idx_to_cluster_array = sch.fcluster(linkage, cluster_distance_threshold, criterion="distance")
    idx = np.argsort(idx_to_cluster_array)

    if not inplace:
        corr = corr.copy()

    if isinstance(corr, pd.DataFrame):
        return corr.iloc[idx, :].T.iloc[idx, :]

    return corr[idx, :][:, idx]


def _prepare_corr_matrix(
    corr: pd.DataFrame,
    max_features: int,
    degree: float,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Prepare and cluster a correlation matrix.

    Parameters
    ----------
    corr : pd.DataFrame
        Correlation matrix.
    max_features : int
        Maximum number of features to display.
    degree : float
        Exponent applied during clustering.

    Returns
    -------
    tuple[pd.DataFrame, list[str], list[str]]
        Clustered correlation matrix, original feature names and shortened
        feature names.
    """
    top_features = compute_top_correlations_features(corr=corr, max_features=max_features)
    corr = _cluster_corr(corr.loc[top_features, top_features], degree=degree)
    list_features = [col for col in corr.columns if col in top_features]

    k = 12
    list_features_shorten = [
        x.replace(x[k + k // 2 : -k + k // 2], "...") if len(x) > 2 * k + 3 else x for x in list_features
    ]
    list_features_shorten = suffix_duplicates(list_features_shorten)
    return corr, list_features, list_features_shorten


def _resolve_style_dict(
    style_dict: dict[str, Any] | None,
    palette_name: str,
) -> dict[str, Any]:
    """
    Resolve the plotting style configuration.

    Parameters
    ----------
    style_dict : dict[str, Any] | None
        User-defined style dictionary.
    palette_name : str
        Name of the color palette.

    Returns
    -------
    dict[str, Any]
        Complete style dictionary.
    """
    if style_dict:
        style_dict_default = {}
        keys = ["dict_title", "init_contrib_colorscale"]
        if any(key not in style_dict for key in keys):
            style_dict_default = define_style(get_palette(palette_name))
        style_dict_default.update(style_dict)
        return style_dict_default

    return define_style(get_palette(palette_name))


def plot_correlations(
    df: pd.DataFrame,
    style_dict: dict[str, Any] | None = None,
    palette_name: str = "default",
    features_dict: dict[str, str] | None = None,
    sample_size: int | None = None,
    max_features: int = 20,
    features_to_hide: Iterable[str] | None = None,
    facet_col: str | None = None,
    how: Literal["phik", "pearson"] = "phik",
    width: int = 900,
    height: int = 500,
    degree: float = 2.5,
    decimals: int = 2,
    file_name: str | None = None,
    auto_open: bool = False,
) -> go.Figure:
    """
    Plot a correlation matrix heatmap.

    Correlations can be computed using either the ``phik`` or ``pearson``
    methods.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset used to compute correlations.
    style_dict : dict[str, Any] | None, default=None
        Custom visualization style configuration.
    palette_name : str, default="default"
        Palette name used when no style configuration is provided.
    features_dict : dict[str, str] | None, default=None
        Mapping between technical feature names and display names.
    sample_size : int | None, default=None
            Maximum number of rows used to compute the correlation matrix.
            If ``None``, no sampling is performed.
    max_features : int, default=20
        Maximum number of features displayed.
    features_to_hide : Iterable[str] | None, default=None
        Features excluded from the correlation matrix.
    facet_col : str | None, default=None
        Column used to split the visualization into several subplots.
    how : Literal['phik', 'pearson'], default="phik"
        Correlation method to use. Supported values are ``phik`` and
        ``pearson``.
    width : int, default=900
        Figure width in pixels.
    height : int, default=500
        Figure height in pixels.
    degree : float, default=2.5
        Exponent applied during clustering.
    decimals : int, default=2
        Number of displayed decimals.
    file_name : str | None, default=None
        Output filename. If ``None``, the figure is not saved.
    auto_open : bool, default=False
        Whether to automatically open the generated plot.

    Returns
    -------
    go.Figure
        Plotly heatmap figure.

    Examples
    --------
    >>> xpl.plot.correlations()
    """

    style_dict_default = _resolve_style_dict(style_dict=style_dict, palette_name=palette_name)

    if features_dict is None:
        features_dict = {}

    if features_to_hide is None:
        features_to_hide = []
    else:
        features_to_hide = list(features_to_hide)

    if sample_size is not None:
        df = df.copy()
        categorical_columns = text_like_columns(df, strict_object=True)
        if facet_col:
            categorical_columns = [col for col in categorical_columns if col != facet_col]

        for col in categorical_columns:
            top_categories = df[col].value_counts().nlargest(200).index
            keep_mask = df[col].isna() | df[col].isin(top_categories)
            if isinstance(df[col].dtype, pd.CategoricalDtype) and "Other" not in df[col].cat.categories:
                df[col] = df[col].cat.add_categories(["Other"])
            df[col] = df[col].where(keep_mask, other="Other")

        if len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=1)

    if facet_col:
        if facet_col not in features_to_hide:
            features_to_hide.append(facet_col)

    hovertemplate = "<b>%{text}<br />Correlation: %{z}</b><extra></extra>"

    list_features: list[str] = []
    if facet_col:
        facet_col_values = sorted(df[facet_col].unique(), reverse=True)
        fig = make_subplots(
            rows=1,
            cols=df[facet_col].nunique(),
            subplot_titles=[f"{t} correlation" for t in facet_col_values],
            horizontal_spacing=0.15,
        )
        # Used for the Shapash report to get train then test set
        for i, col_v in enumerate(facet_col_values):
            df_subset = df[df[facet_col] == col_v]
            corr = compute_corr(df_subset.drop(features_to_hide, axis=1), how)
            corr, list_features, list_features_shorten = _prepare_corr_matrix(
                corr=corr, max_features=max_features, degree=degree
            )

            fig.add_trace(
                go.Heatmap(
                    z=corr.loc[list_features, list_features].round(decimals).values,
                    x=list_features_shorten,
                    y=list_features_shorten,
                    coloraxis="coloraxis",
                    text=[
                        [
                            (f"Feature 1: {features_dict.get(y, y)} <br />Feature 2: {features_dict.get(x, x)}")
                            for x in list_features
                        ]
                        for y in list_features
                    ],
                    hovertemplate=hovertemplate,
                ),
                row=1,
                col=i + 1,
            )

    else:
        corr = compute_corr(df.drop(features_to_hide, axis=1), how)
        corr, list_features, list_features_shorten = _prepare_corr_matrix(
            corr=corr, max_features=max_features, degree=degree
        )

        fig = go.Figure(
            go.Heatmap(
                z=corr.loc[list_features, list_features].round(decimals).values,
                x=list_features_shorten,
                y=list_features_shorten,
                coloraxis="coloraxis",
                text=[
                    [
                        f"Feature 1: {features_dict.get(y, y)} <br />Feature 2: {features_dict.get(x, x)}"
                        for x in list_features
                    ]
                    for y in list_features
                ],
                hovertemplate=hovertemplate,
            )
        )

    title = f"Correlation ({how})"
    if len(list_features) < len(df.drop(features_to_hide, axis=1).columns):
        subtitle = f"Top {len(list_features)} correlations"
        title += f"<span style='font-size: 12px;'><br />{subtitle}</span>"
    dict_t = style_dict_default["dict_title"] | {"text": title, "y": adjust_title_height(height)}

    if corr.min().min() >= 0:
        colorscale = ["rgb(255, 255, 255)"] + style_dict_default["init_contrib_colorscale"][5:-1]
    else:
        colorscale = style_dict_default["init_contrib_colorscale"]

    fig.update_layout(
        coloraxis=dict(colorscale=colorscale),
        showlegend=True,
        title=dict_t,
        width=width,
        height=height,
    )

    fig.update_yaxes(automargin=True)
    fig.update_xaxes(automargin=True)

    if file_name:
        plot(fig, filename=file_name, auto_open=auto_open)

    return fig


def plot_contributions_correlations(
    contributions: pd.DataFrame,
    df: pd.DataFrame | None = None,
    style_dict: dict[str, Any] | None = None,
    palette_name: str = "default",
    features_dict: dict[str, str] | None = None,
    sample_size: int | None = None,
    max_features: int = 20,
    features_to_hide: Iterable[str] | None = None,
    facet_col: str | None = None,
    width: int = 900,
    height: int = 500,
    degree: float = 2.5,
    decimals: int = 2,
    file_name: str | None = None,
    auto_open: bool = False,
) -> go.Figure:
    """
    Plot a contribution-weighted correlation matrix heatmap.

    Correlations are computed from contribution values using
    ``contribution_weighted_corr_matrix``.

    Parameters
    ----------
    contributions : pd.DataFrame
        Contribution values used to compute the correlation matrix.
    df : pd.DataFrame | None, default=None
        DataFrame used for faceting when ``facet_col`` is provided.
        Must share the same index as ``contributions``.
    style_dict : dict[str, Any] | None, default=None
        Custom visualization style configuration.
    palette_name : str, default="default"
        Palette name used when no style configuration is provided.
    features_dict : dict[str, str] | None, default=None
        Mapping between technical feature names and display names.
    sample_size : int | None, default=None
        Maximum number of rows used to compute the correlation matrix.
        If ``None``, no sampling is performed.
    max_features : int, default=20
        Maximum number of features displayed.
    features_to_hide : Iterable[str] | None, default=None
        Features excluded from the correlation matrix.
    facet_col : str | None, default=None
        Column used to split the visualization into several subplots.
    width : int, default=900
        Figure width in pixels.
    height : int, default=500
        Figure height in pixels.
    degree : float, default=2.5
        Exponent applied during clustering.
    decimals : int, default=2
        Number of displayed decimals.
    file_name : str | None, default=None
        Output filename. If ``None``, the figure is not saved.
    auto_open : bool, default=False
        Whether to automatically open the generated plot.

    Returns
    -------
    go.Figure
        Plotly heatmap figure.
    """

    style_dict_default = _resolve_style_dict(style_dict=style_dict, palette_name=palette_name)

    if features_dict is None:
        features_dict = {}

    if features_to_hide is None:
        features_to_hide = []
    else:
        features_to_hide = list(features_to_hide)

    contrib_features_to_hide = [feature for feature in features_to_hide if feature in contributions.columns]

    if df is None:
        df = pd.DataFrame(index=contributions.index)
    else:
        if not contributions.index.isin(df.index).all():
            raise ValueError("df must contain the same index as contributions.")
        df = df.loc[contributions.index].copy()

    if facet_col is not None and facet_col not in df.columns:
        raise ValueError("facet_col must be a column of df.")

    if sample_size is not None and len(contributions) > sample_size:
        sampled_index = contributions.sample(
            n=sample_size,
            random_state=1,
        ).index
        contributions = contributions.loc[sampled_index]
        df = df.loc[sampled_index]

    hovertemplate = "<b>%{text}<br />Correlation: %{z}</b><extra></extra>"

    list_features: list[str] = []
    if facet_col:
        facet_col_values = sorted(df[facet_col].unique(), reverse=True)
        fig = make_subplots(
            rows=1,
            cols=df[facet_col].nunique(),
            subplot_titles=[f"{t} correlation" for t in facet_col_values],
            horizontal_spacing=0.15,
        )
        for i, col_v in enumerate(facet_col_values):
            subset_index = df[df[facet_col] == col_v].index
            corr = contribution_weighted_corr_matrix(
                contributions.loc[subset_index].drop(contrib_features_to_hide, axis=1)
            )
            corr, list_features, list_features_shorten = _prepare_corr_matrix(
                corr=corr, max_features=max_features, degree=degree
            )

            fig.add_trace(
                go.Heatmap(
                    z=corr.loc[list_features, list_features].round(decimals).values,
                    x=list_features_shorten,
                    y=list_features_shorten,
                    coloraxis="coloraxis",
                    text=[
                        [
                            (f"Feature 1: {features_dict.get(y, y)} <br />Feature 2: {features_dict.get(x, x)}")
                            for x in list_features
                        ]
                        for y in list_features
                    ],
                    hovertemplate=hovertemplate,
                ),
                row=1,
                col=i + 1,
            )
    else:
        corr = contribution_weighted_corr_matrix(contributions.drop(contrib_features_to_hide, axis=1))
        corr, list_features, list_features_shorten = _prepare_corr_matrix(
            corr=corr, max_features=max_features, degree=degree
        )

        fig = go.Figure(
            go.Heatmap(
                z=corr.loc[list_features, list_features].round(decimals).values,
                x=list_features_shorten,
                y=list_features_shorten,
                coloraxis="coloraxis",
                text=[
                    [
                        (f"Feature 1: {features_dict.get(y, y)} <br />Feature 2: {features_dict.get(x, x)}")
                        for x in list_features
                    ]
                    for y in list_features
                ],
                hovertemplate=hovertemplate,
            )
        )

    title = "Correlation (contribution-weighted)"
    if len(list_features) < len(contributions.drop(contrib_features_to_hide, axis=1).columns):
        subtitle = f"Top {len(list_features)} correlations"
        title += f"<span style='font-size: 12px;'><br />{subtitle}</span>"
    dict_t = style_dict_default["dict_title"] | {"text": title, "y": adjust_title_height(height)}

    if corr.min().min() >= 0:
        colorscale = ["rgb(255, 255, 255)"] + style_dict_default["init_contrib_colorscale"][5:-1]
    else:
        colorscale = style_dict_default["init_contrib_colorscale"]

    fig.update_layout(
        coloraxis=dict(colorscale=colorscale),
        showlegend=True,
        title=dict_t,
        width=width,
        height=height,
    )

    fig.update_yaxes(automargin=True)
    fig.update_xaxes(automargin=True)

    if file_name:
        plot(fig, filename=file_name, auto_open=auto_open)

    return fig
