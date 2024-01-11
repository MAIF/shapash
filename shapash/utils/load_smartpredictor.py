"""
load_smartpredictor module
"""
from shapash.explainer.smart_predictor import SmartPredictor
from shapash.utils.io import load_pickle


def load_smartpredictor(path):
    """
    load_smartpredictor allows Shapash users to load SmartPredictor Object already saved into a pickle.

    Parameters
    ----------
    path : str
        File path of the pickle file.

    Example
    --------
    >>> predictor = load_smartpredictor('path_to_pkl/predictor.pkl')
    """
    predictor = load_pickle(path)
    if isinstance(predictor, SmartPredictor):
        return predictor
    else:
        raise ValueError(f"{predictor} is not an instance of type SmartPredictor")
