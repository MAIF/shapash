import pandas as pd
import numpy as np
from math import log10, floor


def round_to_1(x):
    """
    round float to 1 significant figure
    Parameters
    ----------
    x : float
        number to round

    Returns
    -------
    int

    """
    if x == 0:
        return 0
    else:
        return round(x, -int(floor(log10(abs(x)))))


def check_row(data, index):
    """
    Identify the row number of datatable for a specific index
    Parameters
    ----------
    data : dash_table.DataTable.data
        data display in datatable
    index : int or str
        index from the dataset to identify

    Returns
    -------
    int:
        row number corresponding to index
    """
    df = pd.DataFrame.from_records(data, index='_index_')
    if np.issubdtype(type(df.index[0]), np.dtype(int).type):
        index = int(index)
    row = df.index.get_loc(index) if index in list(df.index) else None
    return row


def split_filter_part(filter_part):
    """
    Transform dash.datatable filter part into pandas.DataFrame filter (source code : Dash documentation)
    Parameters
    ----------
    filter_part : str
        filter apply on a column of the datatable

    Returns
    -------
    tuple :
        column, operator, value of the filter part

    """
    operators = [['ge ', '>='],
                 ['le ', '<='],
                 ['lt ', '<'],
                 ['gt ', '>'],
                 ['ne ', '!='],
                 ['eq ', '='],
                 ['contains '],
                 ['datestartswith ']]
    for operator_type in operators:
        for operator in operator_type:
            if operator in filter_part:
                name_part, value_part = filter_part.split(operator, 1)
                name = name_part[name_part.find('{') + 1: name_part.rfind('}')]

                value_part = value_part.strip()
                v0 = value_part[0]
                if v0 == value_part[-1] and v0 in ("'", '"', '`'):
                    value = value_part[1: -1].replace('\\' + v0, v0)
                else:
                    try:
                        value = float(value_part)
                    except ValueError:
                        value = value_part

                # word operators need spaces after them in the filter string,
                # but we don't want these later
                return name, operator_type[0].strip(), value

    return [None] * 3


def apply_filter(df, filter_query):
    """
    Apply a filter query from dash.datable to a pandas.DataFrame (source code : Dash documentation)

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
    filtering_expressions = filter_query.split(' && ')
    for filter_part in filtering_expressions:
        col_name, operator, filter_value = split_filter_part(filter_part)
        if operator in ('eq', 'ne', 'lt', 'le', 'gt', 'ge'):
            # these operators match pandas series operator method names
            df = df.loc[getattr(df[col_name], operator)(filter_value)]
        elif operator == 'contains':
            df = df.loc[df[col_name].str.contains(filter_value)]
        elif operator == 'datestartswith':
            # this is a simplification of the front-end filtering logic,
            # only works with complete fields in standard format
            df = df.loc[df[col_name].str.startswith(filter_value)]
    return df

