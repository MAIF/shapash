"""
Data loader module
"""
import os
import json
import pandas as pd
from urllib.request import urlretrieve


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
    current_path = os.path.dirname(os.path.abspath(__file__))
    if dataset == 'house_prices':
        if os.path.isfile(current_path+'/house_prices_dataset.csv') is False:
            github_data_url = 'https://github.com/MAIF/shapash/raw/master/shapash/data/'
            urlretrieve(github_data_url + "house_prices_dataset.csv", filename=current_path + "/house_prices_dataset.csv")
            urlretrieve(github_data_url + "house_prices_labels.json", filename=current_path + "/house_prices_labels.json")
        data_house_prices_path = os.path.join(current_path, "house_prices_dataset.csv")
        dict_house_prices_path = os.path.join(current_path, "house_prices_labels.json")
        data = pd.read_csv(data_house_prices_path, header=0, index_col=0, engine='python')
        with open(dict_house_prices_path, 'r') as openfile2:
            dic = json.load(openfile2)
        return data, dic

    elif dataset == 'titanic':
        if os.path.isfile(current_path + '/titanicdata.csv') is False:
            github_data_url = 'https://github.com/MAIF/shapash/raw/master/shapash/data/'
            urlretrieve(github_data_url + "titanicdata.csv", filename=current_path + "/titanicdata.csv")
            urlretrieve(github_data_url + "titaniclabels.json", filename=current_path + "/titaniclabels.json")
        data_titanic_path = os.path.join(current_path, "titanicdata.csv")
        dict_titanic_path = os.path.join(current_path, 'titaniclabels.json')
        data = pd.read_csv(data_titanic_path, header=0, index_col=0, engine='python')
        with open(dict_titanic_path, 'r') as openfile:
            dic = json.load(openfile)
        return data, dic

    elif dataset == 'telco_customer_churn':
        if os.path.isfile(current_path + '/telco_customer_churn.csv') is False:
            github_data_url = 'https://github.com/IBM/telco-customer-churn-on-icp4d/raw/master/data/'
            urlretrieve(github_data_url + "Telco-Customer-Churn.csv", filename=current_path + "/telco_customer_churn.csv")
        data_telco_path = os.path.join(current_path, "telco_customer_churn.csv")
        data = pd.read_csv(data_telco_path, header=0, index_col=0, engine='python')
        return data

    else:
        raise ValueError("Dataset not found. Check the docstring for available values")
