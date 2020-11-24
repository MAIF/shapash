"""
Check Module
"""

import numpy as np
import pandas as pd
from shapash.utils.transform import preprocessing_tolist, check_transformers
from shapash.utils.columntransformer_backend import columntransformer
import warnings


def check_preprocessing(preprocessing=None):
    """
    Check that all transformation of the preprocessing are supported.

    Parameters
    ----------
    preprocessing: category_encoders, ColumnTransformer, list, dict, optional (default: None)
        The processing apply to the original data
    """
    if preprocessing is not None:
        list_preprocessing = preprocessing_tolist(preprocessing)
        use_ct, use_ce = check_transformers(list_preprocessing)
        return use_ct, use_ce

def check_model(model):
    """
    Check if model has a predict_proba method is a one column dataframe of integer or float
    and if y_pred index matches x_pred index

    Parameters
    ----------
    model: model object
        model used to check the different values of target estimate predict or predict_proba

    Returns
    -------
    string:
        'regression' or 'classification' according to the attributes of the model
    """
    _classes = None
    if hasattr(model, 'predict'):
        if hasattr(model, 'predict_proba') or \
                any(hasattr(model, attrib) for attrib in ['classes_', '_classes']):
            if hasattr(model, '_classes'): _classes = model._classes
            if hasattr(model, 'classes_'): _classes = model.classes_
            if isinstance(_classes, np.ndarray): _classes = _classes.tolist()
            if hasattr(model, 'predict_proba') and _classes == []: _classes = [0, 1]  # catboost binary
            if hasattr(model, 'predict_proba') and _classes is None:
                raise ValueError(
                    "No attribute _classes, classification model not supported"
                )
        if _classes not in (None, []):
            return 'classification', _classes
        else:
            return 'regression', None
    else:
        raise ValueError(
            "No method predict in the specified model. Please, check model parameter"
        )

def check_label_dict(label_dict, case, classes=None):
    """
    Check if label_dict and model _classes match

    Parameters
    ----------
    label_dict: dict
        Dictionary mapping integer labels to domain names (classification - target values).
    case: string
        String that informs if the model used is for classification or regression problem.
    classes: list, None
        List of labels if the model used is for classification problem, None otherwise.
    """
    if label_dict is not None and case == 'classification':
        if set(classes) != set(list(label_dict.keys())):
            raise ValueError(
                "label_dict and don't match: \n" +
                f"label_dict keys: {str(list(label_dict.keys()))}\n" +
                f"Classes model values {str(classes)}"
            )

def check_mask_params(mask_params):
    """
    Check if mask_params given respect the expected format.

    Parameters
    ----------
    mask_params: dict (optional)
        Dictionnary allowing the user to define a apply a filter to summarize the local explainability.
    """
    if not isinstance(mask_params, dict):
        raise ValueError(
            """
            mask_params must be a dict  
            """
        )
    else:
        conform_arguments = ["features_to_hide", "threshold", "positive", "max_contrib"]
        mask_arguments_not_conform = [argument for argument in mask_params.keys()
                                      if argument not in conform_arguments]
        if len(mask_arguments_not_conform) != 0:
            raise ValueError(
            """
            mask_params must only have the following key arguments:
            -feature_to_hide
            -threshold
            -positive
            -max_contrib 
            """
            )

def check_ypred(x=None, ypred=None):
    """
    Check that ypred given has the right shape and expected value.

    Parameters
    ----------
    ypred: pandas.DataFrame (optional)
        User-specified prediction values.
    x: pandas.DataFrame
        Dataset used by the model to perform the prediction (preprocessed or not).
    """
    if ypred is not None:
        if not isinstance(ypred, (pd.DataFrame, pd.Series)):
            raise ValueError("y_pred must be a one column pd.Dataframe or pd.Series.")
        if not ypred.index.equals(x.index):
            raise ValueError("x and y_pred should have the same index.")
        if isinstance(ypred, pd.DataFrame):
            if ypred.shape[1] > 1:
                raise ValueError("y_pred must be a one column pd.Dataframe or pd.Series.")
            if not (ypred.dtypes[0] in [np.float, np.int]):
                raise ValueError("y_pred must contain int or float only")
        if isinstance(ypred, pd.Series):
            if not (ypred.dtype in [np.float, np.int]):
                raise ValueError("y_pred must contain int or float only")
            ypred = ypred.to_frame()
    return ypred

def check_contribution_object(case, classes, contributions):
    """
    Check len of list if _case is "classification"
    Check contributions object type if _case is "regression"
    Check type of contributions and transform into (list of) pd.Dataframe if necessary

    Parameters
    ----------
    case: string
        String that informs if the model used is for classification or regression problem.
    classes: list, None
        List of labels if the model used is for classification problem, None otherwise.
    contributions : pandas.DataFrame, np.ndarray or list
    """
    if case == "regression" and isinstance(contributions, (np.ndarray, pd.DataFrame)) == False:
        raise ValueError(
            """
            Type of contributions parameter specified is not compatible with 
            regression model.
            Please check model and contributions parameters.  
            """
        )
    elif case == "classification":
        if isinstance(contributions, list):
            if len(contributions) != len(classes):
                raise ValueError(
                    """
                    Length of list of contributions parameter is not equal
                    to the number of classes in the target.
                    Please check model and contributions parameters.
                    """
                )
        else:
            raise ValueError(
                """
                Type of contributions parameter specified is not compatible with 
                classification model.
                Please check model and contributions parameters.
                """
            )

def check_preprocessing_options(preprocessing=None):
    """
    Check if preprocessing for ColumnTransformer doesn't have "drop" option
    Parameters
    ----------
    preprocessing: category_encoders, ColumnTransformer, list or dict (optional)
        The processing apply to the original data.
    """
    if preprocessing is not None:
        list_encoding = preprocessing_tolist(preprocessing)
        for enc in list_encoding:
            if str(type(enc)) in columntransformer:
                for options in enc.transformers_:
                    if "drop" in options:
                        raise ValueError("ColumnTransformer remainder 'drop' isn't supported by the SmartPredictor.")

def check_consistency_postprocessing(features_types, columns_dict, postprocessing=None):
    """
    Check that postprocessing parameter has good attributes.
    Check if postprocessing is a dictionnary, and if its parameters are good.

    Parameters
    ----------
    columns_dict: dict
        Dictionary mapping integer column number (in the same order of the trained dataset) to technical feature names.
    features_types: dict
        Dictionnary mapping features with the right types needed.
    postprocessing : dict
        Dictionnary of postprocessing that need to be checked.
    """
    if postprocessing:
        if not isinstance(postprocessing, dict):
            raise ValueError("Postprocessing parameter must be a dictionnary")
        for feature in postprocessing.keys():
            if feature not in features_types.keys():
                raise ValueError("Postprocessing and features_types must have the same features names.")
            if feature not in columns_dict.values():
                raise ValueError("Postprocessing and columns_dict must have the same features names.")

        for key in postprocessing.keys():

            dict_post = postprocessing[key]

            if not isinstance(dict_post, dict):
                raise ValueError(f"{key} values must be a dict")

            if not list(dict_post.keys()) == ['type', 'rule']:
                raise ValueError("Wrong postprocessing keys, you need 'type' and 'rule' keys")

            if not dict_post['type'] in ['prefix', 'suffix', 'transcoding', 'regex', 'case']:
                raise ValueError("Wrong postprocessing method. \n"
                                 "The available methods are: 'prefix', 'suffix', 'transcoding', 'regex', or 'case'")

            if dict_post['type'] == 'case':
                if dict_post['rule'] not in ['lower', 'upper']:
                    raise ValueError("Case modification unknown. Available ones are 'lower', 'upper'.")

                if features_types[key] != "object":
                    raise ValueError(f"Expected string object to modify with upper/lower method in {key} dict")

            if dict_post['type'] == 'regex':
                if not set(dict_post['rule'].keys()) == {'in', 'out'}:
                    raise ValueError(f"Regex modifications for {key} are not possible, the keys in 'rule' dict"
                                     f" must be 'in' and 'out'.")

                if features_types[key] != "object":
                    raise ValueError(f"Expected string object to modify with regex methods in {key} dict")
