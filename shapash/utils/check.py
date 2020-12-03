"""
Check Module
"""

import numpy as np
import pandas as pd
from shapash.utils.category_encoder_backend import no_dummies_category_encoder, supported_category_encoder
from shapash.utils.columntransformer_backend import no_dummies_sklearn, columntransformer, supported_sklearn
from shapash.utils.model import extract_features_model
from shapash.utils.model_synoptic import dict_model_feature
from shapash.utils.transform import preprocessing_tolist, check_transformers
from shapash.utils.columntransformer_backend import columntransformer



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
            if isinstance(ypred.columns[0], (np.int, np.float)):
                ypred.columns = ["ypred"]
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

def check_consistency_model_features(features_dict, model, columns_dict, features_types,
                                     mask_params=None, preprocessing=None, postprocessing=None):
    """
    Check the matching between attributes, features names are same, or include

    Parameters
    ----------
    features_dict: dict
        Dictionary mapping technical feature names to domain names.
    model: model object
        model used to check the different values of target estimate predict_proba
    columns_dict: dict
        Dictionary mapping integer column number (in the same order of the trained dataset) to technical feature names.
    features_types: dict
        Dictionnary mapping features with the right types needed.
    preprocessing: category_encoders, ColumnTransformer, list or dict (optional)
            The processing apply to the original data
    mask_params: dict (optional)
        Dictionnary allowing the user to define a apply a filter to summarize the local explainability.
    postprocessing : dict
        Dictionnary of postprocessing that need to be checked.
    """
    if features_dict is not None:
        if not all(feat in features_types for feat in features_dict):
            raise ValueError("All features of features_dict must be in features_types")

    if set(features_types) != set(columns_dict.values()):
        raise ValueError("features of features_types and model must be the same")

    if mask_params is not None:
        if mask_params['features_to_hide'] is not None:
            if not all(feature in set(features_types) for feature in mask_params['features_to_hide']):
                raise ValueError("All features of mask_params must be in model")

    if preprocessing is not None and str(type(preprocessing)) in (supported_category_encoder, supported_sklearn):
        if not all(feature in set(columns_dict.values()) for feature in set(preprocessing.cols)):
            raise ValueError("All features of preprocessing must be in columns_dict")

    model_features = extract_features_model(model, dict_model_feature[str(type(model))])
    if isinstance(model_features, list):
        if str(type(preprocessing)) in no_dummies_category_encoder:
            if set(columns_dict.values()) != set(model_features):
                raise ValueError("features of columns_dict and model must be the same")

        elif str(type(preprocessing)) in (no_dummies_sklearn, columntransformer):
             if len(set(columns_dict.values())) != len(set(model_features)):
                raise ValueError("length of features of columns_dict and model must be the same")

        elif str(type(preprocessing)) not in (no_dummies_category_encoder, no_dummies_sklearn, columntransformer)\
                and preprocessing is not None:
            raise ValueError("this type of encoder is not supported in SmartPredictor")
    else:
        model_length_features = model_features
        if len(set(columns_dict.values())) != model_length_features:
            raise ValueError("features of columns_dict and model must have the same length")

    if postprocessing:
        if not isinstance(postprocessing, dict):
            raise ValueError("Postprocessing parameter must be a dictionnary")
        for feature in postprocessing.keys():
            if feature not in features_types.keys():
                raise ValueError("Postprocessing and features_types must have the same features names.")
            if feature not in columns_dict.values():
                raise ValueError("Postprocessing and columns_dict must have the same features names.")
        check_postprocessing(features_types, postprocessing)

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

def check_consistency_model_label(columns_dict, label_dict=None):
    """
    Check the matching between attributes, features names are same, or include

    Parameters
    ----------
    columns_dict: dict
        Dictionary mapping integer column number (in the same order of the trained dataset) to technical feature names.
    label_dict: dict (optional)
        Dictionary mapping integer labels to domain names (classification - target values).
    """

    if label_dict is not None:
        if not all(feat in columns_dict for feat in label_dict):
            raise ValueError("All features of label_dict must be in model")

def check_postprocessing(x, postprocessing=None):
    """
    Check that postprocessing parameter has good attributes matching with x dataset or with dict of types of
    the expected data set x

    Parameters
    ----------
    x: pandas.DataFrame, dict
        Dataset x without preprocessing or dictionnary mapping features with the right types needed.
    postprocessing : dict
        Dictionnary of postprocessing that need to be checked.
    """
    if postprocessing:
        if not isinstance(postprocessing, dict):
            raise ValueError("Postprocessing parameter must be a dictionnary")

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

                if isinstance(x, dict):
                    if x[key] != "object":
                        raise ValueError(f"Expected string object to modify with upper/lower method in {key} dict")
                else:
                    if not pd.api.types.is_string_dtype(x[key]):
                        raise ValueError(f"Expected string object to modify with upper/lower method in {key} dict")

            if dict_post['type'] == 'regex':
                if not set(dict_post['rule'].keys()) == {'in', 'out'}:
                    raise ValueError(f"Regex modifications for {key} are not possible, the keys in 'rule' dict"
                                     f" must be 'in' and 'out'.")
                if isinstance(x,dict):
                    if x[key] != "object":
                        raise ValueError(f"Expected string object to modify with regex methods in {key} dict")
                else:
                    if not pd.api.types.is_string_dtype(x[key]):
                        raise ValueError(f"Expected string object to modify with upper/lower method in {key} dict")

def check_features_name(columns_dict, features_dict, features):
    """
    Convert a list of feature names (string) or features ids into features ids.
    Features names can be part of columns_dict or features_dict.

    Parameters
    ----------
    features : List
        List of ints (columns ids) or of strings (business names)
    columns_dict: dict
    Dictionary mapping integer column number to technical feature names.
    features_dict: dict
    Dictionary mapping technical feature names to domain names.

    Returns
    -------
    list of ints
        Columns ids compatible with var_dict
    """
    if all(isinstance(f, int) for f in features):
        features_ids = features

    elif all(isinstance(f, str) for f in features):
        inv_columns_dict = {v: k for k, v in columns_dict.items()}
        inv_features_dict = {v: k for k, v in features_dict.items()}

        if features_dict and all(f in features_dict.values() for f in features):
            columns_list = [inv_features_dict[f] for f in features]
            features_ids = [inv_columns_dict[c] for c in columns_list]
        elif inv_columns_dict and all(f in columns_dict.values() for f in features):
            features_ids = [inv_columns_dict[f] for f in features]
        else:
            raise ValueError(
                'All features must came from the same dict of features (technical names or domain names).'
            )

    else:
        raise ValueError(
            """
            features must be a list of ints (representing ids of columns)
            or a list of string from technical features names or from domain names.
            """
        )
    return features_ids




