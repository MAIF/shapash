"""
Data loader module
"""
import os.path as op
import json
import pandas as pd


def data_loading(dataset):
    """
    data_loading allows shapash user to try the library with small but clear datasets.
    Titanic's or house_prices' reworked data loader 
    from 'titanicdata.csv' and 'house_prices_dataset.csv'
    with well labels in a dictionnary.

    Example
    ----------
    >>> from shapash.data.data_loader import data_loading
    >>> house_df, house_dict = data_loading('house_prices')

    Parameters
    ----------
    dataset : String
        Dataset's name to return.
         - 'titanic'
         - 'house_prices'

    Returns
    -------
    data : pandas.DataFrame
        Dataset required
    dict : Dictionnary
        Columns labels dictionnary associated to the dataset.
    """
    current_path = op.dirname(op.abspath(__file__))
    if dataset == 'house_prices':
        data_house_prices_path = op.join(current_path, "house_prices_dataset.csv")
        dict_house_prices_path = op.join(current_path, "house_prices_labels.json")
        data = pd.read_csv(data_house_prices_path, header=0, index_col=0, engine='python')
        with open(dict_house_prices_path, 'r') as openfile2:
            dic = json.load(openfile2)
    elif dataset == 'titanic':
        data_titanic_path = op.join(current_path, "titanicdata.csv")
        dict_titanic_path = op.join(current_path, 'titaniclabels.json')
        data = pd.read_csv(data_titanic_path, header=0, index_col=0, engine='python')
        with open(dict_titanic_path, 'r') as openfile:
            dic = json.load(openfile)
    else:
        raise ValueError("Dataset not found. Check the docstring for available values")

    return data, dic
