"""
Select Lines Module
"""
def select_lines(dataframe, condition=None):
    """
    Select lines of a pandas.DataFrame based
    on a boolean condition.
    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe used for the query.
    condition : string
        A boolean condition expressed as string.
    Returns
    -------
    list
         list of indices, or lines, to select.
    """
    if condition:
        return dataframe.query(condition).index.values.tolist()
    else:
        return []
