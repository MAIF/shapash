"""
Translate Module
"""


def translate(elements, mapping):
    """
    Map a dictionary to a list of elements.

    Parameters
    ----------
    elements : list
        List of elements to translate.
    mapping : dict
        Dictionary to apply to each elements.

    Returns
    -------
    List
        The list of business names (strings) obtained.
    """
    return [mapping[element] for element in elements]
