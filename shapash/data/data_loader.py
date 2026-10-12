"""
Data loader module
"""

import json
import os
from http.client import HTTPResponse
from pathlib import Path
from typing import Any, Literal
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import urlopen

import pandas as pd


def _safe_urlopen(url: str, **kwargs: Any) -> HTTPResponse:
    """Open an HTTPS URL safely with forwarded request options.

    Parameters
    ----------
    url : str
        URL to open. Only HTTPS schemes are allowed.
    **kwargs : Any
        Keyword arguments forwarded to :func:`urllib.request.urlopen`.

    Returns
    -------
    HTTPResponse
        The response object returned by :func:`urllib.request.urlopen`.

    Raises
    ------
    ValueError
        If the URL scheme is not HTTPS.
    """
    if urlparse(url).scheme != "https":
        raise ValueError(f"Only HTTPS URLs are permitted: {url}")
    return urlopen(url, **kwargs)  # noqa: S310


def _find_file(data_path: str, github_data_url: str, filename: str) -> str:
    """Find a file locally or verify its availability at the remote URL.

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
        Local path or remote URL of the file.

    Raises
    ------
    ConnectionError
        If the file is not available locally and cannot be reached remotely.
    """
    file = os.path.join(data_path, filename)
    if os.path.isfile(file) is False:
        file = github_data_url + filename
        try:
            with _safe_urlopen(file, timeout=10):
                pass
        except URLError as exc:
            raise ConnectionError(f"Internet connection is required to download: {file}") from exc
    return file


def data_loading(
    dataset: Literal["titanic", "house_prices", "telco_customer_churn", "us_car_accident"],
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, str]]:
    """
    Load one of the sample datasets supported by Shapash.

    Example
    ----------
    >>> from shapash.data.data_loader import data_loading
    >>> house_df, house_dict = data_loading('house_prices')

    Parameters
    ----------
    dataset : str
        Dataset name: ``'titanic'``, ``'house_prices'``,
        ``'telco_customer_churn'`` or ``'us_car_accident'``.

    Returns
    -------
    pandas.DataFrame or tuple[pandas.DataFrame, dict[str, str]]
        The dataset as a DataFrame. For ``'titanic'`` and ``'house_prices'``,
        returns a tuple containing the DataFrame and its column-label mapping.

    Raises
    ------
    ValueError
        If ``dataset`` is not one of the supported dataset names.
    """
    data_path = str(Path(__file__).parents[2] / "data")
    if dataset == "house_prices":
        github_data_url = "https://github.com/MAIF/shapash/raw/master/data/"
        data_house_prices_path = _find_file(data_path, github_data_url, "house_prices_dataset.csv")
        dict_house_prices_path = _find_file(data_path, github_data_url, "house_prices_labels.json")
        data = pd.read_csv(data_house_prices_path, header=0, index_col=0, engine="python")
        if github_data_url in dict_house_prices_path:
            with _safe_urlopen(dict_house_prices_path) as openfile:
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
            with _safe_urlopen(dict_titanic_path) as openfile:
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
