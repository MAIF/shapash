"""
Model Module
"""

from inspect import ismethod
from typing import Any, Literal

import numpy as np
import pandas as pd


def extract_features_model(model: Any, model_attribute: list[str]) -> Any:
    """
    Extract model feature metadata, or the model's input feature count.

    Parameters
    ----------
    model : Any
        Model from which to extract feature metadata.
    model_attribute : list[str]
        Attribute path to follow, or ["length"] to retrieve ``n_features_in_``.

    Returns
    -------
    Any
        The requested model feature metadata.
    """
    if model_attribute[0] == "length":
        return model.n_features_in_
    else:
        if ismethod(getattr(model, model_attribute[0])):
            if len(model_attribute) == 1:
                return getattr(model, model_attribute[0])()
            else:
                return extract_features_model(getattr(model, model_attribute[0])(), model_attribute[1:])
        else:
            if len(model_attribute) == 1:
                return getattr(model, model_attribute[0])
            else:
                return extract_features_model(getattr(model, model_attribute[0]), model_attribute[1:])


def predict_proba(model: Any, x_encoded: pd.DataFrame, classes: list[int | float | str]) -> pd.DataFrame:
    """
    Compute class probabilities for each row in ``x_encoded``.

    Parameters
    ----------
    model : Any
        Model expected to provide a ``predict_proba`` method.
    x_encoded : pandas.DataFrame
        Prediction set.
    classes : list[Any]
        Ordered class labels used to name the probability columns.
    Returns
    -------
    pandas.DataFrame
        Predicted probability for each class and row.

    Raises
    ------
    ValueError
        If the model has no ``predict_proba`` method or classes are omitted at runtime.
    """
    if classes is None:
        raise ValueError("classes must be provided to predict class probabilities")

    if hasattr(model, "predict_proba"):
        proba_values = pd.DataFrame(
            model.predict_proba(x_encoded), columns=["class_" + str(x) for x in classes], index=x_encoded.index
        )
    else:
        raise ValueError("model has no predict_proba method")

    return proba_values


def predict(model: Any, x_encoded: pd.DataFrame) -> pd.DataFrame:
    """
    Compute predictions for each row in ``x_encoded``.

    Parameters
    ----------
    model : Any
        Model expected to provide a ``predict`` method.
    x_encoded: pandas.DataFrame
        Observations on which to compute predictions.

    Returns
    -------
    pandas.DataFrame
        One-column DataFrame containing the predictions.
    """
    if hasattr(model, "predict"):
        y_pred = pd.DataFrame(model.predict(x_encoded), columns=["pred"], index=x_encoded.index)
    else:
        raise ValueError("model has no predict method")

    return y_pred


def predict_error(
    y_target: pd.DataFrame | None,
    y_pred: pd.DataFrame | None,
    model_type: Literal["regression", "classification"],
    proba_values: pd.DataFrame | None = None,
    classes: list[int | float | str] | None = None,
) -> pd.DataFrame | None:
    """
    Compute prediction errors for regression or classification.

    For regression:
        - If the target can be zero, absolute error is used:
                error = |y_true - y_pred|
        - Otherwise, relative error is used:
                error = |(y_true - y_pred) / y_true|

    For classification:
        - The error is computed as:
                error = |1 - P(true_class)|
        - The probability of the true class is retrieved using the index:
                col_index = classes.index(label_code)
            where:
              * `classes` is the ordered list of label codes coming from the model
              * `label_code` is the true label from y_target
              * the matching column in `proba_values` corresponds to P(class == label_code)

    Parameters
    ----------
    y_target : pandas.DataFrame or None
        One-column DataFrame containing the ground truth labels, or None.
    y_pred : pandas.DataFrame or None
        One-column DataFrame containing the predicted labels, or None.
    model_type : Literal["regression", "classification"]
        Either "regression" or "classification".
    proba_values : pandas.DataFrame or None, optional
        DataFrame of class probabilities returned by model.predict_proba().
        Each column corresponds to a class, in the same order as in `classes`.
    classes : list[int or float or str] or None, optional
        Ordered list of class label codes (`model.classes_`), used to map the
        true label to the correct probability column when probabilities are supplied.

    Returns
    -------
    pandas.DataFrame
        One-column DataFrame containing the prediction errors, named "_error_", or None
        when either target or predictions are unavailable.

    Raises
    ------
    ValueError
        If class probabilities are provided without classes, or a target label is not
        present in the supplied classes.
    """

    if y_target is None or y_pred is None:
        return None

    # ================= REGRESSION =================
    if model_type == "regression":
        if (y_target == 0).any().iloc[0]:
            prediction_error = abs(y_target.values - y_pred.values)
        else:
            prediction_error = abs((y_target.values - y_pred.values) / y_target.values)

        return pd.DataFrame(prediction_error, index=y_target.index, columns=["_error_"])

    # ================= CLASSIFICATION =================
    elif model_type == "classification":
        if proba_values is None:
            target_values = np.asarray(y_target).reshape(-1)
            prediction_values = np.asarray(y_pred).reshape(-1)
            prediction_error = (target_values != prediction_values).astype(int)
            return pd.DataFrame(prediction_error, index=y_target.index, columns=["_error_"])

        if classes is None:
            raise ValueError("classes must be provided when class probabilities are supplied")

        true_labels = y_target.iloc[:, 0]
        label_to_col = {cls: i for i, cls in enumerate(classes)}
        col_indices = true_labels.map(label_to_col)
        if col_indices.isna().any():
            raise ValueError("Unknown label in y_target")

        proba_true = proba_values.to_numpy()[np.arange(len(proba_values)), col_indices.to_numpy()]

        # Erreur = 1 - proba de la vraie classe
        errors = np.abs(1 - proba_true)

        return pd.DataFrame(errors, index=y_target.index, columns=["_error_"])
