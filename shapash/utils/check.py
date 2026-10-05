import copy
from typing import Any, Literal, cast

import numpy as np
import pandas as pd

from shapash.utils.category_encoder_backend import supported_category_encoder
from shapash.utils.columntransformer_backend import columntransformer, get_list_features_names
from shapash.utils.model import extract_features_model
from shapash.utils.model_synoptic import dict_model_feature
from shapash.utils.transform import check_transformers, preprocessing_tolist


def _is_string_dtype_metadata(dtype_value: Any) -> bool:
    if not isinstance(dtype_value, str):
        return False
    return dtype_value in {"object", "str", "string"} or dtype_value.startswith("string[")


def check_preprocessing(preprocessing: Any | None = None) -> tuple[bool, bool] | None:
    """
    Check that all transformation of the preprocessing are supported.

    Parameters
    ----------
    preprocessing: Any, optional
        The processing apply to the original data

    Returns
    -------
    tuple[bool, bool] or None
        Whether ColumnTransformer and category encoders are used, or None when preprocessing is None.
    """
    if preprocessing is not None:
        list_preprocessing = preprocessing_tolist(preprocessing)
        use_ct, use_ce = check_transformers(list_preprocessing)
        return use_ct, use_ce
    return None


def check_model(model: Any) -> tuple[Literal["regression", "classification"], list[Any] | None]:
    """Determine whether a model supports classification or regression.

    Parameters
    ----------
    model: Any
        Model expected to provide a predict method and, for classification, class metadata.

    Returns
    -------
    tuple[str, list[Any] or None]
        The model type ('regression' or 'classification') and its classes, if it is a classifier.
    """
    _classes = None
    if hasattr(model, "predict"):
        if hasattr(model, "predict_proba") or any(hasattr(model, attrib) for attrib in ["classes_", "_classes"]):
            if hasattr(model, "_classes"):
                _classes = model._classes
            if hasattr(model, "classes_"):
                _classes = model.classes_
            if isinstance(_classes, np.ndarray):
                _classes = _classes.tolist()
            if hasattr(model, "predict_proba") and _classes == []:
                _classes = [0, 1]  # catboost binary
            if hasattr(model, "predict_proba") and _classes is None:
                raise ValueError("No attribute _classes, classification model not supported")
        if _classes not in (None, []):
            return "classification", _classes
        else:
            return "regression", None
    else:
        raise ValueError("No method predict in the specified model. Please, check model parameter")


def check_label_dict(
    label_dict: dict[Any, Any] | None,
    case: Literal["regression", "classification"],
    classes: list[Any] | None = None,
) -> None:
    """
    Check if label_dict and model _classes match

    Parameters
    ----------
    label_dict: dict[Any, Any] or None
        Dictionary mapping integer labels to domain names (classification - target values).
    case: str
        String that informs if the model used is for classification or regression problem.
    classes: list[Any] or None, optional
        List of labels if the model used is for classification problem, None otherwise.
    """
    if label_dict is not None and case == "classification":
        if set(cast(list[Any], classes)) != set(list(label_dict.keys())):
            raise ValueError(
                "label_dict and don't match: \n"
                + f"label_dict keys: {str(list(label_dict.keys()))}\n"
                + f"Classes model values {str(classes)}"
            )


def check_mask_params(mask_params: dict[str, Any]) -> None:
    """
    Check if mask_params given respect the expected format.

    Parameters
    ----------
    mask_params: dict[str, Any]
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
        mask_arguments_not_conform = [argument for argument in mask_params.keys() if argument not in conform_arguments]
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


def check_y(
    x: pd.DataFrame | None = None,
    y: pd.DataFrame | pd.Series | None = None,
    y_name: str = "y_target",
) -> pd.DataFrame | None:
    """
    Check that ypred given has the right shape and expected value.

    Parameters
    ----------
    x: pandas.DataFrame or None, optional
        Dataset used by the model to perform the prediction (preprocessed or not). Required when y is not None.
    y: pandas.DataFrame or pandas.Series or None, optional
        User-specified prediction values. A Series is converted to a one-column DataFrame.
    y_name: str
        Name of y ("y_target" or "y_pred")

    Returns
    -------
    pandas.DataFrame or None
        The validated values, with a Series converted to a DataFrame, or None when y is None.
    """
    if y is not None:
        if not isinstance(y, pd.DataFrame | pd.Series):
            raise ValueError(f"{y_name} must be a one column pd.Dataframe or pd.Series.")
        if not y.index.equals(cast(pd.DataFrame, x).index):
            raise ValueError(f"x and {y_name} should have the same index.")
        if isinstance(y, pd.DataFrame):
            if y.shape[1] > 1:
                raise ValueError(f"{y_name} must be a one column pd.Dataframe or pd.Series.")
            if y.dtypes.iloc[0] not in [float, int, np.int32, np.float32, np.int64, np.float64]:
                raise ValueError(f"{y_name} must contain int or float only")
        if isinstance(y, pd.Series):
            if y.dtype not in [float, int, np.int32, np.float32, np.int64, np.float64]:
                raise ValueError(f"{y_name} must contain int or float only")
            y = y.to_frame()
            if isinstance(y.columns[0], int | float):
                y.columns = [y_name]
    return y


def check_contribution_object(
    case: Literal["regression", "classification"],
    classes: list[Any] | None,
    contributions: np.ndarray | pd.DataFrame | list[Any],
) -> None:
    """Validate the type and number of contribution objects for the model case.

    Parameters
    ----------
    case: str
        String that informs if the model used is for classification or regression problem.
    classes: list[Any] or None
        List of labels if the model used is for classification problem, None otherwise.
    contributions: pandas.DataFrame, numpy.ndarray or list
        Contributions for the model; classification requires one object per class.
    """
    if (case == "regression") and (not isinstance(contributions, np.ndarray | pd.DataFrame)):
        raise ValueError(
            """
            Type of contributions parameter specified is not compatible with
            regression model.
            Please check model and contributions parameters.
            """
        )
    elif case == "classification":
        if isinstance(contributions, list):
            if len(contributions) != len(cast(list[Any], classes)):
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


def check_consistency_model_features(
    features_dict: dict[str, str] | None,
    model: Any,
    columns_dict: dict[int, str],
    features_types: dict[str, str],
    mask_params: dict[str, Any] | None = None,
    preprocessing: Any | None = None,
    postprocessing: dict[str, Any] | None = None,
    list_preprocessing: list[Any] | None = None,
    features_groups: dict[str, list[str]] | None = None,
) -> None:
    """
    Check the matching between attributes, features names are same, or include

    Parameters
    ----------
    features_dict: dict[str, str] or None
        Dictionary mapping technical feature names to domain names.
    model: Any
        model used to check the different values of target estimate predict_proba
    columns_dict: dict[int, str]
        Dictionary mapping integer column number (in the same order of the trained dataset) to technical feature names.
    features_types: dict[str, str]
        Dictionnary mapping features with the right types needed.
    preprocessing: Any, optional
            The processing apply to the original data
    mask_params: dict[str, Any] or None, optional
        Dictionnary allowing the user to define a apply a filter to summarize the local explainability.
    postprocessing: dict[str, Any] or None, optional
        Dictionnary of postprocessing that need to be checked.
    list_preprocessing: list[Any] or None, optional
        List containing all preprocessing steps; used when preprocessing is provided.
    features_groups: dict[str, list[str]] or None, optional
        Mapping of group names to their feature names.
    """
    # Features dict can include additional entries for groups of features.
    # We don't want to check them here as they may not be in other dict
    features_dict = copy.deepcopy(features_dict)
    if features_dict is not None and features_groups is not None:
        for feat in features_groups.keys():
            if feat in features_dict.keys():
                features_dict.pop(feat)

    if features_dict is not None:
        if not all(feat in features_types for feat in features_dict):
            raise ValueError("All features of features_dict must be in features_types")

    if set(features_types) != set(columns_dict.values()):
        raise ValueError("features of features_types and columns_dict must be the same")

    if mask_params is not None:
        if mask_params["features_to_hide"] is not None:
            if not all(feature in set(features_types) for feature in mask_params["features_to_hide"]):
                raise ValueError("All features of mask_params must be in model")

    if preprocessing is not None and str(type(preprocessing)) in (supported_category_encoder):
        if not all(feature in set(columns_dict.values()) for feature in set(preprocessing.cols)):
            raise ValueError("All features of preprocessing must be in columns_dict")

    model_features = extract_features_model(model, dict_model_feature[str(type(model))])
    if isinstance(model_features, list):
        feature_expected_model = model_features
        model_expected = len(set(model_features))
    else:
        feature_expected_model = None
        model_expected = model_features

    if preprocessing is None:
        if isinstance(feature_expected_model, list):
            if set(columns_dict.values()) != set(feature_expected_model):
                columns_dict_feature = [str(feature) for feature in columns_dict.values()]
                if set(columns_dict_feature) != set(feature_expected_model):
                    raise ValueError("Features of columns_dict and model must be the same.")
        else:
            if len(set(columns_dict.values())) != model_expected:
                raise ValueError("Features of columns_dict and model must have the same length")

    if str(type(preprocessing)) in supported_category_encoder and isinstance(feature_expected_model, list):
        if set(cast(Any, preprocessing).feature_names_out_) != set(feature_expected_model):
            raise ValueError(
                """
                                One of features returned by the Category_Encoders preprocessing doesn't
                                match the model's expected features.
                            """
            )
    elif preprocessing is not None:
        if list_preprocessing is None:
            raise ValueError("list_preprocessing is required when preprocessing is provided.")
        feature_encoded = list(get_list_features_names(list_preprocessing, columns_dict))
        if model_expected != len(feature_encoded):
            raise ValueError(
                """
                Number of features returned by the preprocessing step doesn't
                match the model's expected features.
                        """
            )

    if postprocessing:
        if not isinstance(postprocessing, dict):
            raise ValueError("Postprocessing parameter must be a dictionnary")
        for feature in postprocessing.keys():
            if feature not in features_types.keys():
                raise ValueError("Postprocessing and features_types must have the same features names.")
            if feature not in columns_dict.values():
                raise ValueError("Postprocessing and columns_dict must have the same features names.")
        check_postprocessing(features_types, postprocessing)


def check_preprocessing_options(
    columns_dict: dict[int, str],
    features_dict: dict[str, str],
    preprocessing: Any | None = None,
    list_preprocessing: list[Any] | None = None,
) -> dict[str, Any] | None:
    """
    Check if preprocessing for ColumnTransformer doesn't have "drop" option otherwise compute several
    informations to adapt the SmartPredictor's actions

    Parameters
    ----------
    preprocessing: Any, optional
        The processing apply to the original data.
    columns_dict: dict[int, str]
        Dictionary mapping integer column number (in the same order of the trained dataset) to technical feature names.
    features_dict: dict[str, str]
        Dictionary mapping technical feature names to domain names.
    list_preprocessing: list[Any] or None, optional
        list containing all preprocessing.
    Returns
    -------
    dict[str, Any] or None
        None if there isn't drop options in ColumnTransformer otherwise dict of informations to adapt.
    """
    feature_to_drop = list()
    if preprocessing is not None:
        for enc in cast(list[Any], list_preprocessing):
            if str(type(enc)) in columntransformer:
                for options in enc.transformers_:
                    if "drop" in options:
                        feature_to_drop.extend(options[2])

    if len(feature_to_drop) != 0:
        feature_to_drop = [index if isinstance(index, str) else columns_dict[index] for index in feature_to_drop]
        features_dict_op = {key: value for key, value in features_dict.items() if key not in feature_to_drop}

        i = 0
        columns_dict_op = dict()
        for value in columns_dict.values():
            if value not in feature_to_drop:
                columns_dict_op[i] = value
                i += 1

        return {
            "features_to_drop": feature_to_drop,
            "features_dict_op": features_dict_op,
            "columns_dict_op": columns_dict_op,
        }

    else:
        return None


def check_consistency_model_label(columns_dict: dict[Any, Any], label_dict: dict[Any, Any] | None = None) -> None:
    """Check that label dictionary keys are present in columns_dict.

    Parameters
    ----------
    columns_dict: dict[Any, Any]
        Mapping of model column identifiers to feature names.
    label_dict: dict[Any, Any] or None, optional
        Mapping of model label values to domain names.
    """

    if label_dict is not None:
        if not all(feat in columns_dict for feat in label_dict):
            raise ValueError("All features of label_dict must be in model")


def check_postprocessing(
    x: pd.DataFrame | dict[str, str], postprocessing: dict[str, dict[str, Any]] | None = None
) -> None:
    """
    Check that postprocessing parameter has good attributes matching with x dataset or with dict of types of
    the expected data set x

    Parameters
    ----------
    x: pandas.DataFrame or dict[str, str]
        Dataset x without preprocessing or dictionnary mapping features with the right types needed.
    postprocessing: dict[str, dict[str, Any]] or None, optional
        Dictionnary of postprocessing that need to be checked.
    """
    if postprocessing:
        if not isinstance(postprocessing, dict):
            raise ValueError("Postprocessing parameter must be a dictionnary")

        for key in postprocessing.keys():
            dict_post = postprocessing[key]

            if not isinstance(dict_post, dict):
                raise ValueError(f"{key} values must be a dict")

            if list(dict_post.keys()) != ["type", "rule"]:
                raise ValueError("Wrong postprocessing keys, you need 'type' and 'rule' keys")

            if dict_post["type"] not in ["prefix", "suffix", "transcoding", "regex", "case"]:
                raise ValueError(
                    "Wrong postprocessing method. \n"
                    "The available methods are: 'prefix', 'suffix', 'transcoding', 'regex', or 'case'"
                )

            if dict_post["type"] == "case":
                if dict_post["rule"] not in ["lower", "upper"]:
                    raise ValueError("Case modification unknown. Available ones are 'lower', 'upper'.")

                if isinstance(x, dict):
                    if not _is_string_dtype_metadata(x[key]):
                        raise ValueError(
                            f"Expected string dtype metadata (object/str/string/string[...]) "
                            f"to apply upper/lower in {key} dict, got {x[key]!r}"
                        )
                else:
                    if not pd.api.types.is_string_dtype(x[key]):
                        raise ValueError(
                            f"Expected a string dtype to apply upper/lower on column {key}, got {x[key].dtype!r}"
                        )

            if dict_post["type"] == "regex":
                if set(dict_post["rule"].keys()) != {"in", "out"}:
                    raise ValueError(
                        f"Regex modifications for {key} are not possible, the keys in 'rule' dict"
                        f" must be 'in' and 'out'."
                    )
                if isinstance(x, dict):
                    if not _is_string_dtype_metadata(x[key]):
                        raise ValueError(
                            f"Expected string dtype metadata (object/str/string/string[...]) "
                            f"to apply regex methods in {key} dict, got {x[key]!r}"
                        )
                else:
                    if not pd.api.types.is_string_dtype(x[key]):
                        raise ValueError(
                            f"Expected a string dtype to apply regex methods on column {key}, got {x[key].dtype!r}"
                        )


def check_features_name(
    columns_dict: dict[int, str], features_dict: dict[str, str], features: list[int | str]
) -> list[int]:
    """
    Convert a list of feature names (string) or features ids into features ids.
    Features names can be part of columns_dict or features_dict.

    Parameters
    ----------
    columns_dict: dict[int, str]
        Dictionary mapping integer column number to technical feature names.
    features_dict: dict[str, str]
        Dictionary mapping technical feature names to domain names.
    features: list[int or str]
        List of integer column ids or strings containing technical or domain names.

    Returns
    -------
    list[int]
        Columns ids compatible with var_dict
    """
    if all(isinstance(f, int) for f in features):
        features_ids = cast(list[int], features)

    elif all(isinstance(f, str) for f in features):
        feature_names = cast(list[str], features)
        inv_columns_dict = {v: k for k, v in columns_dict.items()}
        inv_features_dict = {v: k for k, v in features_dict.items()}

        if features_dict and all(f in features_dict.values() for f in feature_names):
            columns_list = [inv_features_dict[f] for f in feature_names]
            features_ids = [inv_columns_dict[c] for c in columns_list]
        elif inv_columns_dict and all(f in columns_dict.values() for f in feature_names):
            features_ids = [inv_columns_dict[f] for f in feature_names]
        else:
            raise ValueError("All features must came from the same dict of features (technical names or domain names).")

    else:
        raise ValueError(
            """
            features must be a list of ints (representing ids of columns)
            or a list of string from technical features names or from domain names.
            """
        )
    return features_ids


def check_additional_data(x: pd.DataFrame, additional_data: pd.DataFrame) -> None:
    """Validate that additional_data is a DataFrame with the same index as x.

    Parameters
    ----------
    x: pandas.DataFrame
        Reference dataset.
    additional_data: pandas.DataFrame
        Additional data to validate.
    """
    if not isinstance(additional_data, pd.DataFrame):
        raise ValueError("additional_data must be a pd.Dataframe.")
    if not additional_data.index.equals(x.index):
        raise ValueError("x and additional_data should have the same index.")


def check_columns_order(columns_order: list[str]) -> None:
    """Validate that columns_order is a list of strings.

    Parameters
    ----------
    columns_order: list[str]
        Column names in the desired order.
    """
    if not isinstance(columns_order, list):
        raise ValueError("columns_order must be a list.")
    if not all(isinstance(item, str) for item in columns_order):
        raise ValueError("All elements in columns_order must be strings.")
