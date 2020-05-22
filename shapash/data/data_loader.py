import pandas as pd
import os.path as op
import json

def data_loading(dataset = 'titanic'):
    """
    Titanic's or house_prices' reworked data loader from 'titanicdata.csv' and 'house_prices_dataset.csv'
    with well labels in a dictionnary.

    Parameters
    ----------
    dataset : String (default = 'titanic')
        Dataset's name to return.
         - 'titanic'
         - 'house_prices'

    Returns
    -------
    data : pandas.DataFrame
        Dataset required
    dic : Dictionnary
        Dictionnary associated to the dataset.
    """
    #Path to current file
    dir = op.dirname(op.abspath(__file__))

    if (dataset == 'house_prices'):
        # House prices
        data_house_prices_path = op.join(dir, "../../tutorial/data/house_prices_dataset.csv")
        dict_house_prices_path = op.join(dir, "../../tutorial/data/house_prices_labels.json")

        # data
        data = pd.read_csv(data_house_prices_path, header=0, index_col=0, engine='python')

        # dic
        with open(dict_house_prices_path, 'r') as openfile2:
            dic = json.load(openfile2)

    elif (dataset=='titanic'):
        #Titanic
        data_titanic_path = op.join(dir, "../../tutorial/data/titanicdata.csv")
        dict_titanic_path = op.join(dir, '../../tutorial/data/titaniclabels.json')

        #data
        data = pd.read_csv(data_titanic_path, header=0, index_col=0, engine='python')

        #dic
        with open(dict_titanic_path, 'r') as openfile:
            dic = json.load(openfile)

    else:
        raise ValueError("Dataset not found. Check the docstring for available values")

    return data, dic
