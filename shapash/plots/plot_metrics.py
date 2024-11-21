from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from shapash.style.style_utils import get_palette, get_pyplot_color


def generate_confusion_matrix_plot(
    y_true: Union[np.array, list],
    y_pred: Union[np.array, list],
    colors_dict: Optional[dict] = None,
    width: int = 7,
    height: int = 4,
    palette_name: str = "default",
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
    col_scale = get_pyplot_color(colors=colors_dict["report_confusion_matrix"])
    cmap_gradient = LinearSegmentedColormap.from_list("col_corr", col_scale, N=100)

    df_cm = pd.crosstab(y_true, y_pred, rownames=["Actual"], colnames=["Predicted"])
    fig, ax = plt.subplots(figsize=(width, height))
    sns.heatmap(df_cm, ax=ax, annot=True, cmap=cmap_gradient, fmt="g")
    return fig
