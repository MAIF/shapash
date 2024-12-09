from typing import Optional

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from plotly import graph_objs as go
from plotly.offline import plot
from plotly.subplots import make_subplots

from shapash.manipulation.summarize import compute_corr
from shapash.style.style_utils import define_style, get_palette
from shapash.utils.utils import adjust_title_height, compute_top_correlations_features, suffix_duplicates


def plot_correlations(
    df,
    style_dict: Optional[dict] = None,
    palette_name: str = "default",
    features_dict=None,
    optimized=False,
    max_features=20,
    features_to_hide=None,
    facet_col=None,
    how="phik",
    width=900,
    height=500,
    degree=2.5,
    decimals=2,
    file_name=None,
    auto_open=False,
):
    """
    Correlations matrix heatmap plot.
    The method can use phik or pearson correlations.
    The correlations computed can be changed using the parameter 'how'.
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame for which we want to compute correlations.
    style_dict: dict
        the different styles used in the different outputs of Shapash
    palette_name : str, optional, default="default"
        The name of the color palette to be used if `colors_dict` is not provided.
    features_dict: dict (default: None)
        Dictionary mapping technical feature names to domain names.
    optimized : boolean, optional
        True if we want to potentially accelerate the computation of the correlation matrix by reducing the
        lenght of the data and the number of modalties per columns.
    max_features : int (default: 10)
        Max number of features to show on the matrix.
    features_to_hide : list (optional)
        List of features that will not appear on the graph
    facet_col : str (optional)
        Name of the column used to split the graph in two (or more) plots. One correlation
        subplot will be computed for each value of this column.
    how : str (default: 'phik')
        Correlation method used. 'phik' or 'pearson' are possible values. 'phik' is used by default.
    width : Int (default: 900)
        Plotly figure - layout width
    height : Int (default: 600)
        Plotly figure - layout height
    degree  : int, optional, (default 2.5)
        degree applied on the correlation matrix in order to focus more or less the clustering
        on strong correlated variables
    decimals : int, optional, (default 2)
        number of decimals to plot for correlation values
    file_name: string (optional)
        File name to use to save the plotly bar chart. If None the bar chart will not be saved.
    auto_open: Boolean (optional)
        Indicate whether to open the bar plot or not.
    Returns
    -------
    go.Figure
    Example
    --------
    >>> xpl.plot.correlations()
    """

    def cluster_corr(corr, degree, inplace=False):
        """
        Rearranges the correlation matrix, corr, so that groups of highly
        correlated variables are next to eachother

        Parameters
        ----------
        corr : pandas.DataFrame or numpy.ndarray
            a NxN correlation matrix
        degree  : int
            degree applied on the correlation matrix in order to focus more or less the clustering
            on strong correlated variables
        inplace : bool, optional
            to replace the original correlation matrix by the new one, by default False

        Returns
        -------
        pandas.DataFrame or numpy.ndarray
            a NxN correlation matrix with the columns and rows rearranged
        """

        if corr.shape[0] < 2:
            return corr

        pairwise_distances = sch.distance.pdist(corr**degree)
        linkage = sch.linkage(pairwise_distances, method="complete")
        cluster_distance_threshold = pairwise_distances.max() / 2
        idx_to_cluster_array = sch.fcluster(linkage, cluster_distance_threshold, criterion="distance")
        idx = np.argsort(idx_to_cluster_array)

        if not inplace:
            corr = corr.copy()

        if isinstance(corr, pd.DataFrame):
            return corr.iloc[idx, :].T.iloc[idx, :]

        return corr[idx, :][:, idx]

    # Function to compute correlation matrix and prepare top features
    def prepare_corr_matrix(df_subset):
        corr = compute_corr(df_subset.drop(features_to_hide, axis=1), compute_method)
        top_features = compute_top_correlations_features(corr=corr, max_features=max_features)
        corr = cluster_corr(corr.loc[top_features, top_features], degree=degree)
        list_features = [col for col in corr.columns if col in top_features]

        # Shorten long feature names and handle duplicates
        k = 12
        list_features_shorten = [
            x.replace(x[k + k // 2 : -k + k // 2], "...") if len(x) > 2 * k + 3 else x for x in list_features
        ]
        list_features_shorten = suffix_duplicates(list_features_shorten)
        return corr, list_features, list_features_shorten

    if style_dict:
        style_dict_default = {}
        keys = ["dict_title", "init_contrib_colorscale"]
        if any(key not in style_dict for key in keys):
            style_dict_default = define_style(get_palette(palette_name))
        style_dict_default.update(style_dict)
    else:
        style_dict_default = define_style(get_palette(palette_name))

    if features_dict is None:
        features_dict = {}

    if features_to_hide is None:
        features_to_hide = []

    if optimized:
        categorical_columns = df.select_dtypes(include=["object", "category"]).columns

        for col in categorical_columns:
            top_categories = df[col].value_counts().nlargest(200).index
            df[col] = df[col].where(df[col].isin(top_categories), other="Other")

        if len(df) > 10000:
            df = df.sample(n=10000, random_state=1)

    if facet_col:
        features_to_hide += [facet_col]

    compute_method = how

    hovertemplate = "<b>%{text}<br />Correlation: %{z}</b><extra></extra>"

    list_features = []
    if facet_col:
        facet_col_values = sorted(df[facet_col].unique(), reverse=True)
        fig = make_subplots(
            rows=1,
            cols=df[facet_col].nunique(),
            subplot_titles=[t + " correlation" for t in facet_col_values],
            horizontal_spacing=0.15,
        )
        # Used for the Shapash report to get train then test set
        for i, col_v in enumerate(facet_col_values):
            df_subset = df[df[facet_col] == col_v]
            corr, list_features, list_features_shorten = prepare_corr_matrix(df_subset)

            fig.add_trace(
                go.Heatmap(
                    z=corr.loc[list_features, list_features].round(decimals).values,
                    x=list_features_shorten,
                    y=list_features_shorten,
                    coloraxis="coloraxis",
                    text=[
                        [
                            f"Feature 1: {features_dict.get(y, y)} <br />" f"Feature 2: {features_dict.get(x, x)}"
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
        corr, list_features, list_features_shorten = prepare_corr_matrix(df)

        fig = go.Figure(
            go.Heatmap(
                z=corr.loc[list_features, list_features].round(decimals).values,
                x=list_features_shorten,
                y=list_features_shorten,
                coloraxis="coloraxis",
                text=[
                    [
                        f"Feature 1: {features_dict.get(y, y)} <br />" f"Feature 2: {features_dict.get(x, x)}"
                        for x in list_features
                    ]
                    for y in list_features
                ],
                hovertemplate=hovertemplate,
            )
        )

    title = f"Correlation ({compute_method})"
    if len(list_features) < len(df.drop(features_to_hide, axis=1).columns):
        subtitle = f"Top {len(list_features)} correlations"
        title += f"<span style='font-size: 12px;'><br />{subtitle}</span>"
    dict_t = style_dict_default["dict_title"] | {"text": title, "y": adjust_title_height(height)}

    fig.update_layout(
        coloraxis=dict(colorscale=["rgb(255, 255, 255)"] + style_dict_default["init_contrib_colorscale"][5:-1]),
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
