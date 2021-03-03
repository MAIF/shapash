"""
Model Module
"""
from inspect import ismethod
import pandas as pd

def extract_features_model(model, model_attribute):
    """
    Extract features of models if it's possible,
    If not extract the number features of model
     -------
    model: model object
        model used to check the different values of target estimate predict proba
    model_attribute: String or List
        if model can give features, attributes to access features, if not 'length'
    """
    if model_attribute[0] == 'length':
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


def predict_proba(model, x_init, classes):
    """
    The predict_proba compute the proba values for each x_init row
    Parameters
    -------
    model: model object
        model used to check the different values of target estimate predict proba
    x_init: pandas.DataFrame
        Prediction set.
    classes: list
        List of labels if the model used is for classification problem, None otherwise.
    Returns
    -------
    pandas.DataFrame
            dataset of predicted proba for each label.
    """
    if hasattr(model, 'predict_proba'):
        proba_values = pd.DataFrame(
            model.predict_proba(x_init),
            columns=['class_' + str(x) for x in classes],
            index=x_init.index)
    else:
        raise ValueError("model has no predict_proba method")

    return proba_values


def predict(model, x_init):
    """
    The predict function computes the prediction values for each x_init row

    Parameters
    -------
    model: model object
        model used to perform predictions
    x_init: pandas.DataFrame
        Observations on which to compute predictions.

    Returns
    -------
    pandas.DataFrame
            1-column dataframe containing the predictions.
    """
    if hasattr(model, 'predict'):
        y_pred = pd.DataFrame(model.predict(x_init), columns=['pred'], index=x_init.index)
    else:
        raise ValueError("model has no predict method")

    return y_pred
