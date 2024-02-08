"""
Data loader module
"""
import json
import os
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import pandas as pd


def _find_file(data_path, github_data_url, filename):
    """
    Finds file path on disk if it exists or gets file path on github.

    Parameters
    ----------
    data_path : str
        Data folder path
    github_data_url : str
        Github data url
    filename : str
        Name of the file

    Returns
    -------
    str
        Founded file path.
    """
    file = os.path.join(data_path, filename)
    if os.path.isfile(file) is False:
        file = github_data_url + filename
        try:
            urlopen(file)
        except URLError:
            raise Exception(f"Internet connection is required to download: {file}")
    return file


def data_loading(dataset):
    """
    data_loading allows shapash user to try the library with small but clear datasets.
    Titanic, house_prices or telco_customer_churn data.

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
         - 'telco_customer_churn'

    Returns
    -------
    data : pandas.DataFrame
        Dataset required
    dict : (Dictionnary, Optional)
        If exist, columns labels dictionnary associated to the dataset.
    """
    data_path = str(Path(__file__).parents[2] / "data")
    if dataset == "house_prices":
        github_data_url = "https://github.com/MAIF/shapash/raw/master/data/"
        data_house_prices_path = _find_file(data_path, github_data_url, "house_prices_dataset.csv")
        dict_house_prices_path = _find_file(data_path, github_data_url, "house_prices_labels.json")
        data = pd.read_csv(data_house_prices_path, header=0, index_col=0, engine="python")
        if github_data_url in dict_house_prices_path:
            with urlopen(dict_house_prices_path) as openfile:
                dic = json.load(openfile)
        else:
            with open(dict_house_prices_path) as openfile:
                dic = json.load(openfile)
        return data, dic

    elif dataset == "titanic":
        github_data_url = "https://github.com/MAIF/shapash/raw/master/data/"
        data_titanic_path = _find_file(data_path, github_data_url, "titanicdata.csv")
        dict_titanic_path = _find_file(data_path, github_data_url, "titaniclabels.json")
        data = pd.read_csv(data_titanic_path, header=0, index_col=0, engine="python")
        if github_data_url in dict_titanic_path:
            with urlopen(dict_titanic_path) as openfile:
                dic = json.load(openfile)
        else:
            with open(dict_titanic_path) as openfile:
                dic = json.load(openfile)
        return data, dic

    elif dataset == "telco_customer_churn":
        github_data_url = "https://github.com/IBM/telco-customer-churn-on-icp4d/raw/master/data/"
        data_telco_path = _find_file(data_path, github_data_url, "Telco-Customer-Churn.csv")
        data = pd.read_csv(data_telco_path, header=0, index_col=0, engine="python")
        return data

    elif dataset == "us_car_accident":
        github_data_url = "https://github.com/MAIF/shapash/raw/master/data/"
        data_accidents_path = _find_file(data_path, github_data_url, "US_Accidents_extract.csv")
        data = pd.read_csv(data_accidents_path, header=0, engine="python")
        return data

    else:
        raise ValueError("Dataset not found. Check the docstring for available values")
