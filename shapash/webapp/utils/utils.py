from collections.abc import Collection
from typing import Any, Literal

import pandas as pd
from pandas.api.types import is_any_real_numeric_dtype


def round_to_k(x: float | int, k: int) -> float | int:
    """
    Round float to k significant digits.

    Remark that integers are also rounded: for example, round_to_k(123456789, 3) will return 123000000.

    Parameters
    ----------
    x : float or int
        number to round
    k : int
        the number of significant figures

    Returns
    -------
    float or int

    """
    x = float(x)
    new_x = float(f"{x:.{k}g}")
    return int(new_x) if new_x % 1 == 0 else new_x


class IndexLabels:
    """
    Translate between the explainer's index values and the row keys the webapp sends to the browser.

    A flat index is passed through unchanged. Each entry of a MultiIndex gets a readable string
    label such as ``"FR | Paris | 0"``: Dash serializes a tuple as a JSON list, which pandas cannot
    use to look a row up, and a string also survives the datatable, the index input and plot
    ``customdata`` unchanged.

    Parameters
    ----------
    index : pandas.Index
        Index of the explainer's ``x_init``.
    """

    SEPARATOR = " | "

    def __init__(self, index: pd.Index) -> None:
        self.is_multi = isinstance(index, pd.MultiIndex)
        self._key_by_label: dict[str, tuple[Any, ...]] = {}
        self._label_by_key: dict[tuple[Any, ...], str] = {}
        if self.is_multi:
            keys = list(index.unique())
            labels = [self.SEPARATOR.join(str(level) for level in key) for key in keys]
            if len(set(labels)) != len(labels):
                # A level value contains the separator, or two values print the same:
                # suffix the position so every row keeps a distinct key.
                labels = [f"{label} [{position}]" for position, label in enumerate(labels)]
            self._key_by_label = dict(zip(labels, keys, strict=True))
            self._label_by_key = dict(zip(keys, labels, strict=True))

    def labels(self, index: pd.Index) -> pd.Index | list[str]:
        """
        Return the row keys of ``index`` as shown in the webapp's ``_index_`` column.

        Parameters
        ----------
        index : pandas.Index
            Index values to label, all of them part of the index given at construction.

        Returns
        -------
        pandas.Index or list[str]
            ``index`` itself for a flat index, the string labels for a MultiIndex.
        """
        if not self.is_multi:
            return index
        return [self._label_by_key[key] for key in index]

    def to_label(self, value: Any) -> Any:
        """
        Convert an index value coming from the browser into the webapp row key.

        Parameters
        ----------
        value : Any
            A row key, or a raw index value as found in plot ``customdata`` (a JSON list for a
            MultiIndex entry).

        Returns
        -------
        Any
            The row key, ``None`` if a list matches no MultiIndex entry, or ``value`` unchanged
            for a flat index.
        """
        if self.is_multi and isinstance(value, list | tuple):
            return self._label_by_key.get(tuple(value))
        return value

    def to_index(self, value: Any) -> Any:
        """
        Convert a webapp row key into the explainer's index value.

        Parameters
        ----------
        value : Any
            A row key from the ``_index_`` column or the index input.

        Returns
        -------
        Any
            The MultiIndex tuple for a known label, or ``value`` unchanged otherwise.
        """
        if self.is_multi:
            if isinstance(value, str) and value in self._key_by_label:
                return self._key_by_label[value]
            if isinstance(value, list):
                return tuple(value)
        return value

    def to_index_list(self, values: list | None) -> list | None:
        """
        Convert a list of webapp row keys into explainer index values.

        Parameters
        ----------
        values : list or None
            Row keys, typically from ``get_indexes_from_datatable``.

        Returns
        -------
        list or None
            The converted values, or ``None`` when ``values`` is ``None``.
        """
        if values is None or not self.is_multi:
            return values
        return [self.to_index(value) for value in values]


def get_index_level_columns(index: pd.Index, taken: Collection[str]) -> list[str]:
    """
    Name one webapp column per MultiIndex level, so each level can be shown and filtered on its own.

    A level is named ``index_<name>``, or ``index_<position>`` when it has no name, so the column
    name depends only on the level and all level columns sort together. In the unlikely case that
    this name is already a column of the table, the level position is appended until it is unique.

    Parameters
    ----------
    index : pandas.Index
        Index of the explainer's ``x_init``.
    taken : Collection[str]
        Column names already used by the table.

    Returns
    -------
    list[str]
        One column name per level, in level order; empty for a flat index.
    """
    if not isinstance(index, pd.MultiIndex):
        return []
    used = set(taken)
    columns = []
    for position, name in enumerate(index.names):
        column = f"index_{position if name is None else name}"
        while column in used:
            column = f"{column}_{position}"
        used.add(column)
        columns.append(column)
    return columns


def get_index_type(data: pd.DataFrame) -> Literal["number", "text"]:
    """
    Identify the type of the dataframe index.

    Parameters
    ----------
    data : pandas.DataFrame
        Dataset of the features

    Returns
    -------
    Literal["number", "text"]
        Whether the dataset index is numeric or textual
    """
    if is_any_real_numeric_dtype(data.index):
        return "number"
    else:
        return "text"


def check_row(data: list[dict[str, Any]], index: int | str | None) -> int | None:
    """
    Identify the row number of datatable for a specific index.

    Parameters
    ----------
    data : list[dict[str, Any]]
        data display in datatable
    index : int or str or None
        index from the dataset to identify

    Returns
    -------
    int:
        row number corresponding to index
    """
    if index is not None:
        df = pd.DataFrame.from_records(data, index="_index_")
        row = df.index.get_loc(index) if index in list(df.index) else None
    else:
        row = None
    return row


def split_filter_part(
    filter_part: str,
) -> tuple[str | None, str | None, str | float | None] | list[None]:
    """
    Transform dash.datatable filter part into pandas.DataFrame filter (source code : Dash documentation).

    Parameters
    ----------
    filter_part : str
        filter apply on a column of the datatable

    Returns
    -------
    tuple[str | None, str | None, str | float | None] or list[None]
        Column, operator and value of the filter part; returns a list of three None values
        when no supported operator is found

    """
    operators = [
        ["ge ", ">="],
        ["le ", "<="],
        ["lt ", "<"],
        ["gt ", ">"],
        ["ne ", "!="],
        ["eq ", "="],
        ["contains "],
        ["datestartswith "],
    ]
    for operator_type in operators:
        for operator in operator_type:
            if operator in filter_part:
                name_part, value_part = filter_part.split(operator, 1)
                name = name_part[name_part.find("{") + 1 : name_part.rfind("}")]

                value_part = value_part.strip()
                v0 = value_part[0]
                if v0 == value_part[-1] and v0 in ("'", '"', "`"):
                    value: str | float = value_part[1:-1].replace("\\" + v0, v0)
                else:
                    try:
                        value = float(value_part)
                    except ValueError:
                        value = value_part

                # word operators need spaces after them in the filter string,
                # but we don't want these later
                return name, operator_type[0].strip(), value

    return [None] * 3


def apply_filter(df: pd.DataFrame, filter_query: str) -> pd.DataFrame:
    """
    Apply a filter query from dash.datable to a pandas.DataFrame (source code : Dash documentation).

    Parameters
    ----------
    df : pandas.DataFrame
        dataFrame to be filtered
    filter_query : dcc.datatable.filter_query
        query from dcc.datatable to apply to the DataFrame

    Returns
    -------
    pandas.DataFrame

    """
    filtering_expressions = filter_query.split(" && ")
    for filter_part in filtering_expressions:
        col_name, operator, filter_value = split_filter_part(filter_part)
        if operator in ("eq", "ne", "lt", "le", "gt", "ge"):
            # these operators match pandas series operator method names
            df = df.loc[getattr(df[col_name], operator)(filter_value)]
        elif operator == "contains":
            df = df.loc[df[col_name].str.contains(filter_value)]
        elif operator == "datestartswith":
            # this is a simplification of the front-end filtering logic,
            # only works with complete fields in standard format
            df = df.loc[df[col_name].str.startswith(filter_value)]
    return df
