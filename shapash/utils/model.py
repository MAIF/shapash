"""
Model Module
"""

from inspect import ismethod

import numpy as np
import pandas as pd


def extract_features_model(model, model_attribute):
    """
    Extract features of models if it's possible,
    If not extract the number features of model
    -------
    model: model object
        model used to check the different values of target estimate predict proba
    model_attribute: String or list
        if model can give features, attributes to access features, if not 'length'
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


def predict_proba(model, x_encoded, classes):
    """
    The predict_proba compute the proba values for each x_encoded row
    Parameters
    -------
    model: model object
        model used to check the different values of target estimate predict proba
    x_encoded: pandas.DataFrame
        Prediction set.
    classes: list
        List of labels if the model used is for classification problem, None otherwise.
    Returns
    -------
    pandas.DataFrame
            dataset of predicted proba for each label.
    """
    if hasattr(model, "predict_proba"):
        proba_values = pd.DataFrame(
            model.predict_proba(x_encoded), columns=["class_" + str(x) for x in classes], index=x_encoded.index
        )
    else:
        raise ValueError("model has no predict_proba method")

    return proba_values


def predict(model, x_encoded):
    """
    The predict function computes the prediction values for each x_encoded row

    Parameters
    -------
    model: model object
        model used to perform predictions
    x_encoded: pandas.DataFrame
        Observations on which to compute predictions.

    Returns
    -------
    pandas.DataFrame
            1-column dataframe containing the predictions.
    """
    if hasattr(model, "predict"):
        y_pred = pd.DataFrame(model.predict(x_encoded), columns=["pred"], index=x_encoded.index)
    else:
        raise ValueError("model has no predict method")

    return y_pred


def predict_error(y_target, y_pred, model_type, proba_values=None, classes=None):
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
              * `proba_values.iloc[:, col_index]` corresponds to P(class == label_code)

    Parameters
    ----------
    y_target : pandas.DataFrame
        One-column DataFrame containing the ground truth labels.
    y_pred : pandas.DataFrame
        One-column DataFrame containing the predicted labels.
    model_type : str
        Either "regression" or "classification".
    proba_values : pandas.DataFrame, optional
        DataFrame of class probabilities returned by model.predict_proba().
        Each column corresponds to a class, in the same order as in `classes`.
    classes : list, optional
        Ordered list of class label codes (`model.classes_`), used to map the
        true label to the correct probability column.

    Returns
    -------
    pandas.DataFrame
        One-column DataFrame containing the prediction errors, named "_error_".
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
            prediction_error = (y_target.values != y_pred.values).astype(int)
            return pd.DataFrame(prediction_error, index=y_target.index, columns=["_error_"])

        # classes = order of model.classes_
        true_labels = y_target.iloc[:, 0]
        label_to_col = {cls: i for i, cls in enumerate(classes)}

        try:
            col_indices = true_labels.map(label_to_col)
        except KeyError as err:
            raise ValueError(f"Unknown label in y_target: {err}") from err

        proba_true = proba_values.to_numpy()[np.arange(len(proba_values)), col_indices.to_numpy()]

        # Erreur = 1 - proba de la vraie classe
        errors = np.abs(1 - proba_true)

        return pd.DataFrame(errors, index=y_target.index, columns=["_error_"])
