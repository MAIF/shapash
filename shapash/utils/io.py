"""
IO module
"""
import pickle
import yaml

def save_pickle(obj, path, protocol=pickle.HIGHEST_PROTOCOL):
    """
    Save any python Object in pickle file
    Parameters
    ----------
    obj : any Python Object
    path : str
        File path where the pickled object will be stored.
    protocol : int
        Int which indicates which protocol should be used by the pickler,
        default HIGHEST_PROTOCOL
    """

    if not isinstance(path, str):
        raise ValueError(
            """
            path parameter must be a string
            """
        )

    if not isinstance(protocol, int):
        raise ValueError(
            """
            protocol parameter must be an integer
            """
        )
    with open(path, "wb") as file:
        pickle.dump(obj, file, protocol=protocol)


def load_pickle(path):
    """
    load any pickle file
    Parameters
    ----------
    path : str
        File path where the pickled object is stored.
    Returns
    -------
    object that pickle file contains
    """

    if not isinstance(path, str):
        raise ValueError(
            """
            path parameter must be a string
            """
        )

    with open(path, "rb") as file:
        pklobj = pickle.load(file)

    return pklobj


def load_yaml(path):
    """
    Loads a yaml file

    Parameters
    ----------
    path : str
        File path where the yaml file is stored.
    Returns
    -------
    d : dict
        Python dict containing the parsed yaml file.
    """
    if not isinstance(path, str):
        raise ValueError(
            """
            path parameter must be a string
            """
        )

    with open(path, "r") as f:
        d = yaml.load(f)

    return d
