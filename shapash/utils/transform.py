"""
Transform Module
"""
import category_encoders as ce

def inverse_transform(x_pred, preprocessing=None):
    """
    Reverse transformation of a DataFrame
    giving a category encoders transformer.

    Parameters
    ----------
    x_pred : pandas.DataFrame
        Prediction set.
    preprocessing : object, optional (default: None)
        A single transformer, from category_encoders

    Returns
    -------
    pandas.Dataframe
        The dataframe before preprocessing.
    """
    if preprocessing is None:
        return x_pred
    else:
        if isinstance(preprocessing, (ce.OrdinalEncoder, ce.OneHotEncoder, ce.BaseNEncoder, ce.BinaryEncoder)):
            return preprocessing.inverse_transform(x_pred).fillna('unknow')
        else:
            raise Exception(f'Encoder {type(preprocessing)} is not supported.')
