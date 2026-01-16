"""
Utils is a group of function for the library
"""

import math
import socket

import numpy as np
import pandas as pd

from shapash.explainer.multi_decorator import MultiDecorator
from shapash.explainer.smart_state import SmartState


def adjust_title_height(figure_height=500):
    """
    Adjust the height of the title according to height of the figure

    Parameters
    ----------
    figure_height : int
        height of the figure

    Returns
    -------
    int
        height of the title
    """

    return 1 - 0.1 * 500 / figure_height


def suffix_duplicates(lst):
    """
    Adds suffixes (_2, _3, ...) to non-unique elements in a list to make them unique.

    Args:
        lst (list): The input list of elements (strings) which may contain duplicates.

    Returns:
        list: A new list where non-unique elements have suffixes to ensure uniqueness.

    Example:
        Input: ["feature1", "feature2", "feature1", "feature2", "feature3"]
        Output: ["feature1", "feature2", "feature1_2", "feature2_2", "feature3"]
    """

    seen = {}
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


def get_host_name():
    """
    Get the url of the current host
    Returns
    -------
    String
        host name
    """
    return socket.gethostname()


def inclusion(first_x, second_x):
    """
    Check if a list is included in another.

    Parameters
    ----------
    first_x : list
        List to evaluate.
    second_x : list
        Reference list to compare with.

    Returns
    -------
    bool
        True if first_x is contained in second_x.
    """
    return all(elem in second_x for elem in first_x)


def within_dict(list_param, dict_param):
    """
    Check if a list is included in either dict keys or dict values.

    Parameters
    ----------
    list_param : list
        List to evaluate.
    dict_param : dict
        Reference dictionary to compare.
    """
    return inclusion(list_param, dict_param.keys()) or inclusion(list_param, dict_param.values())


def is_nested_list(object_param):
    """
    Check if object is a nested list or not.

    Parameters
    ----------
    object_param : object
        Any object to check.

    Returns
    -------
    Bool
        True if the object is a nested list, False otherwise.
    """
    return any(isinstance(elem, list) for elem in object_param)


def add_line_break(value, nbchar, maxlen=150):
    """
    adding line break in string if necessary

    Parameters
    ----------
    value : string or oither type
        if string to check in order to add line break
    nbchar : int
        number of characters before line break
    maxlen : int
        number of characters before truncation

    Returns
    -------
    string
        original text + line break
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

        new_string = "".join(sum(zip(input_word, final_sep + [""]), ())[:-1]) + last_char
        return new_string
    else:
        return value


def truncate_str(text, maxlen=40):
    """
    truncate a string

    Parameters
    ----------
    text : string
        string to check in order to add line break
    maxlen : int
        number of characters before truncation

    Returns
    -------
    string
        truncated text
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


def compute_digit_number(value, significant_digits: int = 4):
    """
    return int, number of digits to display

    Parameters
    ----------
    value : float
        can be the gap between percentiles
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


def tuning_round_digit(values: pd.DataFrame, quantile=(0.25, 0.75)):
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


def add_text(text_list, sep):
    """
    return int, number of digits to display

    Parameters
    ----------
    text_list : list
        list of text elements to concat
    sep: str
        separatator

    Returns
    -------
    int
        number of digits
    """
    clean_list = [x for x in text_list if x not in ["", None]]
    return sep.join(clean_list)


def maximum_difference_sort_value(contributions):
    """
    Auxiliary function to sort the contributions for the compare_plot.
    Returns the value of the maximum difference between values in contributions[0].

    Parameters
    ----------
    contributions: list
        list containing 2 elements:
        a Numpy.ndarray of contributions of the indexes compared, and the features' names.

    Returns
    -------
    value_max_difference : float
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


def compute_sorted_variables_interactions_list_indices(interaction_values):
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
    interaction_contrib_sorted_indices : list
        List containing all pairs of indices in descending order of most important interactions.
    """
    tmp = np.abs(interaction_values).sum(0)
    for i in range(tmp.shape[0]):
        tmp[i, i:] = 0

    interaction_contrib_sorted_indices = np.dstack(np.unravel_index(np.argsort(tmp.ravel(), kind="stable"), tmp.shape))[
        0
    ][::-1]
    return interaction_contrib_sorted_indices


def get_project_root():
    """
    Returns project root absolute path.
    """
    from pathlib import Path

    current_path = Path(__file__)

    return current_path.parent.parent.parent.resolve()


def compute_top_correlations_features(corr: pd.DataFrame, max_features: int) -> list:
    """
    Returns the max_features features having top correlations.

    Parameters
    ----------
    corr: pd.DataFrame
    max_features : int

    Returns
    -------
    list
    """
    sorted_corr = corr.abs().unstack().sort_values(kind="quicksort")[::-1]
    set_features = set()
    i = 0
    while len(set_features) < max_features and i < len(sorted_corr):
        if sorted_corr.index[i][0] != sorted_corr.index[i][1]:
            set_features.add(sorted_corr.index[i][0])
            # Last iteration can add one more feature otherwise
            if len(set_features) != max_features:
                set_features.add(sorted_corr.index[i][1])
        i += 1
    return list(set_features)


def choose_state(contributions):
    """
    Select implementation of the smart explainer. Typically check if it is a
    multi-class problem, in which case the implementation should be adapted
    to lists of contributions.

    Parameters
    ----------
    contributions : object
        Local contributions. Could also be a list of local contributions.

    Returns
    -------
    object
        SmartState or SmartMultiState, depending on the nature of the input.
    """
    if isinstance(contributions, list):
        return MultiDecorator(SmartState())
    else:
        return SmartState()


def convert_string_to_int_keys(input_dict: dict) -> dict:
    """
    Returns the dict with integer keys instead of string keys

    Parameters
    ----------
    input_dict: dict

    Returns
    -------
    dict
    """
    return {int(k): v for k, v in input_dict.items()}


def tuning_colorscale(
    init_colorscale,
    values,
    keep_quantile=None,
    quantile_linearization=False,
):
    """
    Adjust the color scale based on the distribution of points.

    This function modifies the color scale used for visualization according to
    the distribution of the provided values. Optionally, it can keep only a specified
    central quantile range to exclude extreme values and focus on the core distribution.

    Parameters
    ----------
    init_colorscale : list
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
        A tuple containing:
        - color_scale : list of tuples (normalized position, color)
        - cmin : float, minimum value used for color scaling
        - cmax : float, maximum value used for color scaling
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
        return color_scale, unique_value, unique_value

    # Case 2: Number of unique values matches number of colors
    if nunique in [2, n_colors]:
        cmin, cmax = min(unique_vals), max(unique_vals)
        positions = np.linspace(0, 1, n_colors)
        color_scale = [(float(pos), col) for pos, col in zip(positions, init_colorscale)]
        return color_scale, cmin, cmax

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

    color_scale = [(float(pos), col) for pos, col in zip(positions, init_colorscale)]
    return color_scale, cmin, cmax


def top_contributors(series: pd.Series, threshold: float = 0.9) -> list:
    """
    Returns the list of names (index values) that cumulatively contribute up to a given threshold of the total.

    Parameters:
    - series (pd.Series): A pandas Series sorted in ascending order, with names as the index.
    - threshold (float): Cumulative contribution threshold (between 0 and 1).

    Returns:
    - list: List of names contributing up to the threshold.
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
