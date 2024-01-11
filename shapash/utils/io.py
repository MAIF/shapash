"""
IO module
"""
import pickle

try:
    import yaml

    _is_yaml_available = True
except ImportError:
    _is_yaml_available = False


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


def load_yml(path):
    """
    Loads a yml file

    Parameters
    ----------
    path : str
        File path where the yml file is stored.
    Returns
    -------
    d : dict
        Python dict containing the parsed yml file.
    """
    if _is_yaml_available is False:
        raise ModuleNotFoundError('Please install PyYAML using "pip install pyyaml" command.')

    if not isinstance(path, str):
        raise ValueError(
            """
            path parameter must be a string
            """
        )

    with open(path) as f:
        d = yaml.full_load(f)

    return d
