"""
load_smartpredictor module
"""
from shapash.explainer.smart_explainer import SmartPredictor
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
    dict_to_load = load_pickle(path)
    if isinstance(dict_to_load, dict):
        predictor = SmartPredictor(features_dict=dict_to_load['features_dict'], model=dict_to_load['model'],
                                   columns_dict=dict_to_load['columns_dict'], explainer=dict_to_load['explainer'],
                                   features_types=dict_to_load['features_types'], label_dict=dict_to_load['label_dict'],
                                   preprocessing=dict_to_load['preprocessing'],
                                   postprocessing=dict_to_load['postprocessing'])
        return predictor
    else:
        raise ValueError(
            "pickle file must contain dictionary"
        )
