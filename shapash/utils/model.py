"""
Modele Module
"""
import pandas as pd

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