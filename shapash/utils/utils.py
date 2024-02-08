"""
Utils is a group of function for the library
"""
import math
import socket

import numpy as np
import pandas as pd

from shapash.explainer.multi_decorator import MultiDecorator
from shapash.explainer.smart_state import SmartState


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


def add_line_break(text, nbchar, maxlen=150):
    """
    adding line break in string if necessary

    Parameters
    ----------
    text : string
        string to check in order to add line break
    nbchar : int
        number of characters before line break
    maxlen : int
        number of characters before truncation

    Returns
    -------
    string
        original text + line break
    """
    if isinstance(text, str):
        length = 0
        tot_length = 0
        input_word = text.split()
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
        return text


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


def compute_digit_number(value):
    """
    return int, number of digits to display

    Parameters
    ----------
    value : float
        can be the gap between percentiles

    Returns
    -------
    int
        number of digits
    """
    # fix for 0 value
    if value == 0:
        first_nz = 1
    else:
        first_nz = int(math.log10(abs(value)))
    digit = abs(min(3, first_nz) - 3)
    return digit


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
