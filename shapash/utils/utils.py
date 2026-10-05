"""
Utils is a group of function for the library
"""

import math
import socket
from collections.abc import Hashable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from shapash.explainer.multi_decorator import MultiDecorator
from shapash.explainer.smart_state import SmartState


def adjust_title_height(figure_height: int | float = 500) -> float:
    """
    Adjust the height of the title according to height of the figure

    Parameters
    ----------
    figure_height : int or float, optional
        height of the figure

    Returns
    -------
    float
        height of the title
    """

    return 1 - 0.1 * 500 / figure_height


def suffix_duplicates(lst: list[str]) -> list[str]:
    """
    Adds suffixes (_2, _3, ...) to non-unique elements in a list to make them unique.

    Parameters
    ----------
    lst : list of str
        The input list of strings which may contain duplicates.

    Returns
    -------
    list of str
        A new list where non-unique elements have suffixes to ensure uniqueness.

    Examples
    --------
        Input: ["feature1", "feature2", "feature1", "feature2", "feature3"]
        Output: ["feature1", "feature2", "feature1_2", "feature2_2", "feature3"]
    """

    seen: dict[str, int] = {}
    result = []

    for item in lst:
        if item in seen:
            # If the item has been seen before, increment its count and add a suffix
            seen[item] += 1
            new_item = f"{item}_{seen[item] + 1}"
        else:
            # If the item is seen for the first time, add it without a suffix
            seen[item] = 0
            new_item = item

        result.append(new_item)

    return result


def get_host_name() -> str:
    """
    Get the hostname of the current host.
    Returns
    -------
    str
        Hostname.
    """
    return socket.gethostname()


def inclusion(first_x: Iterable[Any], second_x: Iterable[Any]) -> bool:
    """
    Check if a list is included in another.

    Parameters
    ----------
    first_x : Iterable[Any]
        List to evaluate.
    second_x : Iterable[Any]
        Reference list to compare with.

    Returns
    -------
    bool
        True if first_x is contained in second_x.
    """
    return all(elem in second_x for elem in first_x)


def within_dict(list_param: Iterable[Any], dict_param: Mapping[Any, Any]) -> bool:
    """
    Check if a list is included in either dict keys or dict values.

    Parameters
    ----------
    list_param : Iterable[Any]
        List to evaluate.
    dict_param : Mapping[Any, Any]
        Reference dictionary to compare.

    Returns
    -------
    bool
        True if all elements are contained in the dictionary keys or values.
    """
    return inclusion(list_param, dict_param.keys()) or inclusion(list_param, dict_param.values())


def is_nested_list(object_param: Iterable[Any]) -> bool:
    """
    Check if object is a nested list or not.

    Parameters
    ----------
    object_param : Iterable[Any]
        Iterable to check.

    Returns
    -------
    Bool
        True if the object is a nested list, False otherwise.
    """
    return any(isinstance(elem, list) for elem in object_param)


def add_line_break(value: Any, nbchar: int, maxlen: int = 150) -> Any:
    """
    adding line break in string if necessary

    Parameters
    ----------
    value : Any
        Text to format; non-string values are returned unchanged.
    nbchar : int
        number of characters before line break
    maxlen : int
        number of characters before truncation

    Returns
    -------
    Any
        Formatted string, or the original value when it is not a string.
    """
    if isinstance(value, str):
        length = 0
        tot_length = 0
        input_word = value.split()
        final_sep = []
        for w in input_word[:-1]:
            length = length + len(w)
            tot_length = tot_length + len(w)
            if tot_length <= maxlen:
                if length >= nbchar:
                    length = 0
                    final_sep.append("<br />")
                else:
                    final_sep.append(" ")
        if len(final_sep) == len(input_word) - 1:
            last_char = ""
        else:
            last_char = "..."

        new_string = "".join(sum(zip(input_word, final_sep + [""], strict=False), ())[:-1]) + last_char
        return new_string
    else:
        return value


def truncate_str(text: Any, maxlen: int = 40) -> Any:
    """
    truncate a string

    Parameters
    ----------
    text : Any
        Text to truncate; non-string values are returned unchanged.
    maxlen : int
        number of characters before truncation

    Returns
    -------
    Any
        Truncated string, or the original value when it is not a string.
    """
    if isinstance(text, str) and len(text) > maxlen:
        tot_length = 0
        input_words = text.split()
        output_words = []
        for word in input_words[:-1]:
            tot_length = tot_length + len(word)
            if tot_length <= maxlen:
                output_words.append(word)

        text = " ".join(output_words)
        if len(input_words) > len(output_words):
            text = text + "..."
    return text


def compute_digit_number(value: int | float | np.number | np.ndarray, significant_digits: int = 4) -> int:
    """
    return int, number of digits to display

    Parameters
    ----------
    value : int, float, numpy.number, or numpy.ndarray
        Numeric value, which can be the gap between percentiles.
    significant_digits : int, optional, default=4
        Fixed number of significant digits to display.

    Returns
    -------
    int
        number of digits
    """
    if isinstance(value, np.ndarray):
        scalar_value = value.item()
    else:
        scalar_value = value

    # fix for 0 value
    if scalar_value == 0:
        first_nz = 1
    else:
        first_nz = math.ceil(math.log10(abs(scalar_value)))
    digit = abs(min(significant_digits, first_nz) - significant_digits)
    return digit


def tuning_round_digit(values: pd.DataFrame, quantile: Sequence[float] = (0.25, 0.75)) -> int:
    """
    return int, number of digits to display

    Parameters
    ----------
    values : pd.DataFrame
        one-column DataFrame containing the values to analyze
    quantile : tuple, optional, default=(0.25, 0.75)
        quantiles to compute the gap

    Returns
    -------
    int
        number of digits
    """
    desc_df = values.describe(percentiles=quantile)
    perc1, perc2 = list(desc_df.loc[[str(int(p * 100)) + "%" for p in quantile]].values)
    p_diff = perc2 - perc1
    return compute_digit_number(p_diff)


def add_text(text_list: Iterable[str | None], sep: str) -> str:
    """
    Concatenate non-empty text elements.

    Parameters
    ----------
    text_list : iterable of str or None
        Text elements to concatenate; empty strings and None values are skipped.
    sep : str
        Separator used between elements.

    Returns
    -------
    str
        Concatenated text.
    """
    clean_list = [x for x in text_list if x not in ["", None]]
    return sep.join(clean_list)


def maximum_difference_sort_value(contributions: Sequence[Sequence[Any]]) -> int | float | np.number:
    """
    Auxiliary function to sort the contributions for the compare_plot.
    Returns the value of the maximum difference between values in contributions[0].

    Parameters
    ----------
    contributions : sequence
        Container whose first element holds the contributions to compare and whose second element holds feature names.

    Returns
    -------
    value_max_difference : int, float, or numpy.number
        Value of the maximum difference contribution.
    """
    if len(contributions[0]) <= 1:
        max_difference = contributions[0][0]
    else:
        max_difference = max(
            [
                abs(contrib_i - contrib_j)
                for i, contrib_i in enumerate(contributions[0])
                for j, contrib_j in enumerate(contributions[0])
                if i <= j
            ]
        )
    return max_difference


def compute_sorted_variables_interactions_list_indices(interaction_values: np.ndarray) -> np.ndarray:
    """
    Returns the sorted interactions as a list of pairs of indices.
    Computes the (absolute) sum of all contributions of each pair of variables in a 2D matrix.
    Then returns the list of all unique pairs of indices of the sorted values in descending order.

    Parameters
    ----------
    interaction_values : np.ndarray
        Numpy array of shape (# samples x # features x # features) containing all interactions for each sample.


    Returns
    -------
    np.ndarray
        Array containing pairs of indices in descending order of interaction importance.
    """
    tmp = np.abs(interaction_values).sum(0)
    for i in range(tmp.shape[0]):
        tmp[i, i:] = 0

    interaction_contrib_sorted_indices = np.dstack(np.unravel_index(np.argsort(tmp.ravel(), kind="stable"), tmp.shape))[
        0
    ][::-1]
    return interaction_contrib_sorted_indices


def get_project_root() -> Path:
    """
    Returns project root absolute path.
    """

    current_path = Path(__file__)

    return current_path.parent.parent.parent.resolve()


def compute_top_correlations_features(corr: pd.DataFrame, max_features: int) -> list[Hashable]:
    """
    Returns the max_features features having top correlations.

    Parameters
    ----------
    corr: pd.DataFrame
    max_features : int

    Returns
    -------
    list[Hashable]
        Feature labels with the highest correlations.
    """
    sorted_corr = corr.abs().unstack().sort_values(kind="quicksort")[::-1]
    set_features: set[Hashable] = set()
    i = 0
    while len(set_features) < max_features and i < len(sorted_corr):
        if sorted_corr.index[i][0] != sorted_corr.index[i][1]:
            set_features.add(sorted_corr.index[i][0])
            # Last iteration can add one more feature otherwise
            if len(set_features) != max_features:
                set_features.add(sorted_corr.index[i][1])
        i += 1
    return list(set_features)


def choose_state(contributions: Any) -> SmartState | MultiDecorator:
    """
    Select implementation of the smart explainer. Typically check if it is a
    multi-class problem, in which case the implementation should be adapted
    to lists of contributions.

    Parameters
    ----------
    contributions : Any
        Local contributions. Could also be a list of local contributions.

    Returns
    -------
    SmartState or MultiDecorator
        State implementation selected from the nature of the input.
    """
    if isinstance(contributions, list):
        return MultiDecorator(SmartState())
    else:
        return SmartState()


def convert_string_to_int_keys(input_dict: dict[str, Any]) -> dict[int, Any]:
    """
    Returns the dict with integer keys instead of string keys

    Parameters
    ----------
    input_dict : dict[str, Any]

    Returns
    -------
    dict[int, Any]
    """
    return {int(k): v for k, v in input_dict.items()}


def tuning_colorscale(
    init_colorscale: Sequence[str],
    values: pd.DataFrame,
    keep_quantile: tuple[float, float] | None = None,
    quantile_linearization: bool = False,
) -> tuple[list[tuple[float, str]], float, float]:
    """
    Adjust the color scale based on the distribution of points.

    This function modifies the color scale used for visualization according to
    the distribution of the provided values. Optionally, it can keep only a specified
    central quantile range to exclude extreme values and focus on the core distribution.

    Parameters
    ----------
    init_colorscale : sequence of str
        A list of colors defining the base color scale.
    values : pd.DataFrame
        A one-column DataFrame containing the values for which quantiles need to be calculated.
    keep_quantile : tuple of float or None, optional
        Tuple (low, high) defining the lower and upper quantiles to **keep** in the distribution.
        - If None: the full range of data is used.
        - Example: (0.05, 0.95) keeps the central 90% (removes bottom 5% and top 5%).
        - Example: (0.1, 0.9) keeps the central 80%.
    quantile_linearization : bool, optional, default=False
        If True, force the colorscale positions to be linearly spaced between 0 and 1,
        while still computing colors from quantiles. This prevents the distribution-driven
        non-linear spacing of colors.

    Returns
    -------
    tuple
        A tuple containing the color scale as (normalized position, color) pairs,
        and the minimum and maximum values used for color scaling.
    """
    # Extract the first column of values
    data = values.iloc[:, 0]

    # Initialize variables for min and max values
    cmin, cmax = None, None
    unique_vals = sorted(data.unique())
    nunique = len(unique_vals)
    n_colors = len(init_colorscale)

    # Case 1: All values identical
    if nunique == 1:
        unique_value = unique_vals[0]
        color_scale = [(i / (n_colors - 1), c) for i, c in enumerate(init_colorscale)]
        return color_scale, float(unique_value), float(unique_value)

    # Case 2: Number of unique values matches number of colors
    if nunique in [2, n_colors]:
        cmin, cmax = min(unique_vals), max(unique_vals)
        positions = np.linspace(0, 1, n_colors)
        color_scale = [(float(pos), col) for pos, col in zip(positions, init_colorscale, strict=False)]
        return color_scale, float(cmin), float(cmax)

    # Case 3: Filter based on quantile range if requested
    if keep_quantile is not None:
        if not (0 <= keep_quantile[0] < keep_quantile[1] <= 1):
            raise ValueError("keep_quantile must be a tuple (low, high) with 0 <= low < high <= 1.")
        lower_quantile = data.quantile(keep_quantile[0])
        upper_quantile = data.quantile(keep_quantile[1])
        data_tmp = data[(data >= lower_quantile) & (data <= upper_quantile)]
        # Only keep filtered data if it's meaningful
        if (len(data_tmp) > 20) and (data_tmp.nunique() > 1):
            data = data_tmp
        cmin, cmax = data.min(), data.max()

    # Quantiles used to "sample" the distribution for the colors
    quantile_values = data.quantile(np.linspace(0, 1, n_colors)).to_numpy()

    # If cmin/cmax not set (keep_quantile is None), use full-range bounds
    if cmin is None or cmax is None:
        cmin, cmax = float(data.min()), float(data.max())

    # Build colorscale positions
    if quantile_linearization:
        # Linear positions: 0..1 regardless of distribution
        positions = np.linspace(0, 1, n_colors)
    else:
        # Distribution-aware positions: normalize quantile values into 0..1
        min_q, max_q = float(np.min(quantile_values)), float(np.max(quantile_values))
        if max_q == min_q:
            positions = np.linspace(0, 1, n_colors)
        else:
            positions = (quantile_values - min_q) / (max_q - min_q)

    color_scale = [(float(pos), col) for pos, col in zip(positions, init_colorscale, strict=False)]
    return color_scale, float(cmin), float(cmax)


def top_contributors(series: pd.Series, threshold: float = 0.9) -> list[Hashable]:
    """
    Returns the list of names (index values) that cumulatively contribute up to a given threshold of the total.

    Parameters
    ----------
    series : pandas.Series
        A pandas Series sorted in ascending order, with names as the index.
    threshold : float
        Cumulative contribution threshold (between 0 and 1).

    Returns
    -------
    list of Hashable
        Index values contributing up to the threshold.
    """
    # Check if the series is sorted in ascending order
    if not series.is_monotonic_increasing:
        series = series.sort_values(ascending=True)

    # Reverse the series to start from the highest contributors
    series_desc = series[::-1]

    # Compute the cumulative percentage of the total
    cumulative_ratio = series_desc.cumsum() / series_desc.sum()

    # Select entries where cumulative sum is below or equal to the threshold
    mask = cumulative_ratio <= threshold

    # Include the next item that may slightly exceed the threshold
    if not mask.all():
        mask.iloc[mask.sum()] = True

    # Return the list of names (index values)
    return series_desc[mask].index.tolist()
