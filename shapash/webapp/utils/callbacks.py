import datetime
from typing import Optional, Tuple

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
from dash import dcc, html


def select_data_from_prediction_picking(round_dataframe: pd.DataFrame, selected_data: dict) -> pd.DataFrame:
    """Create a subset dataframe from the prediction picking graph selection.

    Parameters
    ----------
    round_dataframe : pd.DataFrame
        Data to sample
    selected_data : dict
        Selected sample in the prediction picking graph

    Returns
    -------
    pd.DataFrame
        Subset dataframe
    """
    row_ids = []
    for p in selected_data["points"]:
        row_ids.append(p["customdata"])
    df = round_dataframe.loc[row_ids]

    return df


def select_data_from_str_filters(
    df: pd.DataFrame,
    feature_id: list,
    id_str_modality: list,
    val_feature: list,
    val_str_modality: list,
) -> pd.DataFrame:
    """Create a subset dataframe from filters.

    Parameters
    ----------
    df : pd.DataFrame
        Data to sample
    feature_id : list
        features ids
    id_str_modality : list
        string features ids
    val_feature : list
        features names
    val_str_modality : list
        string modalities selected

    Returns
    -------
    pd.DataFrame
        Subset dataframe
    """
    # get list of ID
    str_id = [id_str_modality[i]["index"] for i in range(len(id_str_modality))]
    # If there is some filters
    if len(str_id) > 0:
        for i in range(len(feature_id)):
            if feature_id[i] in str_id:
                position = np.where(np.array(str_id) == feature_id[i])[0][0]
                if (position is not None) & (val_str_modality[position] is not None):
                    df = df[df[val_feature[i]].isin(val_str_modality[position])]

    return df


def select_data_from_bool_filters(
    df: pd.DataFrame,
    feature_id: list,
    id_bool_modality: list,
    val_feature: list,
    val_bool_modality: list,
) -> pd.DataFrame:
    """Create a subset dataframe from filters.

    Parameters
    ----------
    df : pd.DataFrame
        Data to sample
    feature_id : list
        features ids
    id_bool_modality : list
        boolean features ids
    val_feature : list
        features names
    val_bool_modality : list
        boolean modalities selected

    Returns
    -------
    pd.DataFrame
        Subset dataframe
    """
    # get list of ID
    bool_id = [id_bool_modality[i]["index"] for i in range(len(id_bool_modality))]
    # If there is some filters
    if len(bool_id) > 0:
        for i in range(len(feature_id)):
            if feature_id[i] in bool_id:
                position = np.where(np.array(bool_id) == feature_id[i])[0][0]
                if (position is not None) & (val_bool_modality[position] is not None):
                    df = df[df[val_feature[i]] == val_bool_modality[position]]

    return df


def select_data_from_numeric_filters(
    df: pd.DataFrame,
    feature_id: list,
    id_lower_modality: list,
    val_feature: list,
    val_lower_modality: list,
    val_upper_modality: list,
) -> pd.DataFrame:
    """Create a subset dataframe from filters.

    Parameters
    ----------
    df : pd.DataFrame
        Data to sample
    feature_id : list
        features ids
    id_lower_modality : list
        numeric features ids
    val_feature : list
        features names
    val_lower_modality : list
        lower values of numeric filter
    val_upper_modality : list
        upper values of numeric filter

    Returns
    -------
    pd.DataFrame
        Subset dataframe
    """
    # get list of ID
    lower_id = [id_lower_modality[i]["index"] for i in range(len(id_lower_modality))]
    # If there is some filters
    if len(lower_id) > 0:
        for i in range(len(feature_id)):
            if feature_id[i] in lower_id:
                position = np.where(np.array(lower_id) == feature_id[i])[0][0]
                if (
                    (position is not None)
                    & (val_lower_modality[position] is not None)
                    & (val_upper_modality[position] is not None)
                ):
                    if val_lower_modality[position] < val_upper_modality[position]:
                        df = df[
                            (df[val_feature[i]] >= val_lower_modality[position])
                            & (df[val_feature[i]] <= val_upper_modality[position])
                        ]

    return df


def select_data_from_date_filters(
    df: pd.DataFrame,
    feature_id: list,
    id_date: list,
    val_feature: list,
    start_date: list,
    end_date: list,
) -> pd.DataFrame:
    """Create a subset dataframe from filters.

    Parameters
    ----------
    round_dataframe : pd.DataFrame
        Data to sample
    id_feature : list
        features ids
    id_date : list
        date features ids
    val_feature : list
        features names
    start_date : list
        start dates selected
    end_date : list
        end dates selected

    Returns
    -------
    pd.DataFrame
        Subset dataframe
    """
    # get list of ID
    date_id = [id_date[i]["index"] for i in range(len(id_date))]
    # If there is some filters
    if len(date_id) > 0:
        for i in range(len(feature_id)):
            if feature_id[i] in date_id:
                position = np.where(np.array(date_id) == feature_id[i])[0][0]
                if (position is not None) & (start_date[position] < end_date[position]):
                    df = df[((df[val_feature[i]] >= start_date[position]) & (df[val_feature[i]] <= end_date[position]))]

    return df


def get_feature_from_clicked_data(click_data: dict) -> str:
    """Get the feature name from the feature importance graph click data.

    Parameters
    ----------
    click_data : dict
        Feature importance graph click data

    Returns
    -------
    str
        Selected feature
    """
    selected_feature = click_data["points"][0]["label"].replace("<b>", "").replace("</b>", "")
    return selected_feature


def get_feature_from_features_groups(selected_feature: Optional[str], features_groups: dict) -> Optional[str]:
    """Get the group feature name of the selected feature.

    Parameters
    ----------
    selected_feature : Optional[str]
        Selected feature
    features_groups : dict
        Groups names and corresponding list of features

    Returns
    -------
    Optional[str]
        Group feature name or selected feature if not in a group.
    """
    list_sub_features = [f for group_features in features_groups.values() for f in group_features]
    if selected_feature in list_sub_features:
        for k, v in features_groups.items():
            if selected_feature in v:
                selected_feature = k
    return selected_feature


def get_group_name(selected_feature: Optional[str], features_groups: Optional[dict]) -> Optional[str]:
    """Get the group feature name if the selected feature is one of the groups.

    Parameters
    ----------
    selected_feature : Optional[str]
        Selected feature
    features_groups : Optional[dict]
        Groups names and corresponding list of features

    Returns
    -------
    Optional[str]
        Group feature name
    """
    group_name = (
        selected_feature if (features_groups is not None and selected_feature in features_groups.keys()) else None
    )
    return group_name


def get_indexes_from_datatable(data: list, list_index: Optional[list] = None) -> Optional[list]:
    """Get the indexes of the data. If list_index is given and is the same length than
    the indexes, there is no subset selected.

    Parameters
    ----------
    data : list
        Data from the table
    list_index : Optional[list], optional
        Default index list to compare the subset with, by default None

    Returns
    -------
    Optional[list]
        Indexes of the data
    """
    indexes = [d["_index_"] for d in data]
    if list_index is not None and (len(indexes) == len(list_index) or len(indexes) == 0):
        indexes = None
    return indexes


def update_click_data_on_subset_changes(click_data: dict) -> dict:
    """Update click data on subset changes to always correspond to the feature selector graph.

    Parameters
    ----------
    click_data : dict
        Feature importance click data

    Returns
    -------
    dict
        Updated feature importance click data
    """
    point = click_data["points"][0]
    point["curveNumber"] = 0
    click_data = {"points": [point]}
    return click_data


def get_figure_zoom(click_zoom: int) -> bool:
    """Get figure zoom from n_clicks

    Parameters
    ----------
    click_zoom : int
        Number of clicks on zoom button

    Returns
    -------
    bool
        zoom active or not
    """
    click = 2 if click_zoom is None else click_zoom
    if click % 2 == 0:
        zoom_active = False
    else:
        zoom_active = True
    return zoom_active


def get_feature_contributions_sign_to_show(positive: list, negative: list) -> Optional[bool]:
    """Get the feature contributions sign to show on plot.

    Parameters
    ----------
    positive : list
        Click on positive contributions
    negative : list
        Click on negative contributions

    Returns
    -------
    Optional[bool]
        Sign to show on plot
    """
    if positive == [1]:
        sign = None if negative == [1] else True
    else:
        sign = False if negative == [1] else None
    return sign


def update_features_to_display(features: int, nb_columns: int, value: int) -> Tuple[int, int, dict]:
    """Update features to display slider.

    Parameters
    ----------
    features : int
        Number of features to plot from the settings
    nb_columns : int
        Number of columns in the data
    value : int
        Number of columns to plot

    Returns
    -------
    Tuple[int, int, dict]
        Number of columns to plot, Number max of columns to plot, Marks in the slider
    """
    max_value = min(features, nb_columns)
    if max_value % 5 == 0:
        nb_marks = min(int(max_value // 5), 10)
    elif max_value % 4 == 0:
        nb_marks = min(int(max_value // 4), 10)
    elif max_value % 3 == 0:
        nb_marks = min(int(max_value // 3), 10)
    elif max_value % 7 == 0:
        nb_marks = min(int(max_value // 7), 10)
    else:
        nb_marks = 2
    marks = {
        f"{round(max_value * feat / nb_marks)}": f"{round(max_value * feat / nb_marks)}"
        for feat in range(1, nb_marks + 1)
    }
    marks["1"] = "1"
    if max_value < value:
        value = max_value

    return value, max_value, marks


def get_id_card_features(data: list, selected: int, special_cols: list, features_dict: dict) -> pd.DataFrame:
    """Get the features of the selected index for the identity card.

    Parameters
    ----------
    data : list
        Data from the table
    selected : int
        Row number of the selected index
    special_cols : list
        Sepcial columns about the index, the prediction...
    features_dict : dict
        Dictionary mapping technical feature names to domain names

    Returns
    -------
    pd.DataFrame
        Dataframe of the features
    """
    selected_row = pd.DataFrame([data[selected]], index=["feature_value"]).T
    selected_row["feature_name"] = selected_row.index.map(lambda x: x if x in special_cols else features_dict[x])
    return selected_row


def get_id_card_contrib(
    data: dict, index: int, features_dict: dict, columns_dict: dict, label_num: int = None
) -> pd.DataFrame:
    """Get the contributions of the selected index for the identity card.

    Parameters
    ----------
    data : dict
        Data from the smart explainer
    index : int
        Index selected
    features_dict : dict
        Dictionary mapping technical feature names to domain names
    columns_dict : dict
        Dictionary mapping integer column number to technical feature names
    label_num : int, optional
        Label num, by default None

    Returns
    -------
    pd.DataFrame
        Dataframe of the contributions
    """
    if label_num is not None:
        contrib = data["contrib_sorted"][label_num].loc[index, :].values
        var_dict = data["var_dict"][label_num].loc[index, :].values
    else:
        contrib = data["contrib_sorted"].loc[index, :].values
        var_dict = data["var_dict"].loc[index, :].values

    var_dict = [features_dict[columns_dict[x]] for x in var_dict]
    selected_contrib = pd.DataFrame([var_dict, contrib], index=["feature_name", "feature_contrib"]).T
    selected_contrib["feature_contrib"] = selected_contrib["feature_contrib"].apply(lambda x: round(x, 4))

    return selected_contrib


def create_id_card_data(
    selected_row: pd.DataFrame,
    selected_contrib: pd.DataFrame,
    sort_by: str,
    order: bool,
    special_cols: list,
    additional_features_dict: dict,
) -> pd.DataFrame:
    """Merge and sort features and contributions dataframes for the identity card.

    Parameters
    ----------
    selected_row : pd.DataFrame
        Dataframe of the features
    selected_contrib : pd.DataFrame
        Dataframe of the contributions
    sort_by : str
        Column to sort by
    order : bool
        Ascending or descending order
    special_cols : list
        Sepcial columns about the index, the prediction...
    additional_features_dict : dict
        Dictionary mapping technical feature names to domain names for additional data

    Returns
    -------
    pd.DataFrame
        Dataframe of the data for the identity card
    """
    selected_data = selected_row.merge(selected_contrib, how="left", on="feature_name")
    selected_data.index = selected_row.index
    selected_data = pd.concat(
        [
            selected_data.loc[special_cols],
            selected_data.drop(index=special_cols + list(additional_features_dict.keys())).sort_values(
                sort_by, ascending=order
            ),
            selected_data.loc[list(additional_features_dict.keys())].sort_values(sort_by, ascending=order),
        ]
    )

    return selected_data


def create_id_card_layout(selected_data: pd.DataFrame, additional_features_dict: dict) -> list:
    """Create the layout of the identity card

    Parameters
    ----------
    selected_data : pd.DataFrame
        Dataframe of the data for the identity card
    additional_features_dict : dict
        Dictionary mapping technical feature names to domain names for additional data

    Returns
    -------
    list
        Layout of the identity card
    """
    children = []
    for _, row in selected_data.iterrows():
        label_style = (
            {"fontWeight": "bold", "font-style": "italic"}
            if row["feature_name"] in additional_features_dict.values()
            else {"fontWeight": "bold"}
        )
        children.append(
            dbc.Row(
                [
                    dbc.Col(dbc.Label(row["feature_name"]), width=3, style=label_style),
                    dbc.Col(dbc.Label(row["feature_value"]), width=5, className="id_card_solid"),
                    dbc.Col(width=1),
                    dbc.Col(
                        dbc.Row(
                            dbc.Label(format(row["feature_contrib"], ".4f"), width="auto", style={"padding-top": 0}),
                            justify="end",
                        ),
                        width=2,
                        className="id_card_solid",
                    )
                    if not np.isnan(row["feature_contrib"])
                    else None,
                ]
            )
        )

    return children


def get_feature_filter_options(dataframe: pd.DataFrame, features_dict: dict, special_cols: list) -> list:
    """Get the columns names options for the filter.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Dataframe
    features_dict : dict
        Dictionary mapping technical feature names to domain names
    special_cols : list
        Sepcial columns about the index, the prediction...

    Returns
    -------
    list
        Options for the filter
    """
    # We use domain name for feature name
    dict_name = [features_dict[i] for i in dataframe.drop(special_cols, axis=1).columns]
    dict_id = [i for i in dataframe.drop(special_cols, axis=1).columns]
    # Create dataframe to sort it by feature_name
    df_feature_name = pd.DataFrame({"feature_name": dict_name, "feature_id": dict_id})
    df_feature_name = df_feature_name.sort_values(by="feature_name").reset_index(drop=True)
    # Options are sorted by feature_name
    options = [{"label": i, "value": i} for i in special_cols] + [
        {"label": df_feature_name.loc[i, "feature_name"], "value": df_feature_name.loc[i, "feature_id"]}
        for i in range(len(df_feature_name))
    ]

    return options


def create_dropdown_feature_filter(n_clicks_add: Optional[int], options: list) -> html.Div:
    """Create a new dropdown for the filter feature selection.

    Parameters
    ----------
    n_clicks_add : Optional[int]
        Number of clicks on the add filter button
    options : list
        Options for the selection

    Returns
    -------
    html.Div
        Div containing the dropdown
    """
    # ID index definition
    if n_clicks_add is None:
        index_id = 0
    else:
        index_id = n_clicks_add
    # Appending a dropdown block to 'dropdowns_container'children
    subset_filter = html.Div(
        id={"type": "bloc_div", "index": index_id},
        children=[
            html.Div(
                [
                    html.Br(),
                    # div which will contains label
                    html.Div(
                        id={"type": "dynamic-output-label", "index": index_id},
                    ),
                ]
            ),
            html.Div(
                [
                    # div with dopdown button to select feature to filter
                    html.Div(
                        dcc.Dropdown(
                            id={"type": "var_dropdown", "index": index_id}, options=options, placeholder="Variable"
                        ),
                        style={"width": "30%"},
                    ),
                    # div which will contains modalities
                    html.Div(id={"type": "dynamic-output", "index": index_id}, style={"width": "50%"}),
                    # Button to delete bloc
                    dbc.Button(
                        id={"type": "del_dropdown_button", "index": index_id}, children="X", color="warning", size="sm"
                    ),
                ],
                style={"display": "flex"},
            ),
        ],
    )

    return subset_filter


def create_filter_modalities_selection(value: str, id: dict, round_dataframe: pd.DataFrame) -> html.Div:
    """Create the modalities filter according to the feature type.

    Parameters
    ----------
    value : str
        feature name
    id : dict
        id of the filter
    round_dataframe : pd.DataFrame
        Dataframe

    Returns
    -------
    html.Div
        Div containing the modalities selection options
    """
    if type(round_dataframe[value].iloc[0]) == bool:
        new_element = html.Div(
            dcc.RadioItems(
                [{"label": val, "value": val} for val in round_dataframe[value].unique()],
                id={"type": "dynamic-bool", "index": id["index"]},
                value=round_dataframe[value].iloc[0],
                inline=False,
            ),
            style={"width": "65%", "margin-left": "20px"},
        )
    elif (type(round_dataframe[value].iloc[0]) == str) | (
        (type(round_dataframe[value].iloc[0]) == np.int64) & (len(round_dataframe[value].unique()) <= 20)
    ):
        new_element = html.Div(
            dcc.Dropdown(
                id={"type": "dynamic-str", "index": id["index"]},
                options=[{"label": i, "value": i} for i in np.sort(round_dataframe[value].unique())],
                multi=True,
            ),
            style={"width": "65%", "margin-left": "20px"},
        )
    elif (type(round_dataframe[value].iloc[0]) is pd.Timestamp) | (
        type(round_dataframe[value].iloc[0]) is datetime.datetime
    ):
        new_element = (
            html.Div(
                dcc.DatePickerRange(
                    id={"type": "dynamic-date", "index": id["index"]},
                    min_date_allowed=round_dataframe[value].min(),
                    max_date_allowed=round_dataframe[value].max(),
                    start_date=round_dataframe[value].min(),
                    end_date=round_dataframe[value].max(),
                ),
                style={"width": "65%", "margin-left": "20px"},
            ),
        )
    else:
        lower_value = 0
        upper_value = 0
        new_element = html.Div(
            [
                dcc.Input(
                    id={"type": "lower", "index": id["index"]},
                    value=lower_value,
                    type="number",
                    style={"width": "60px"},
                ),
                " <= {} in [{}, {}]<= ".format(value, round_dataframe[value].min(), round_dataframe[value].max()),
                dcc.Input(
                    id={"type": "upper", "index": id["index"]},
                    value=upper_value,
                    type="number",
                    style={"width": "60px"},
                ),
            ],
            style={"margin-left": "20px"},
        )

    return new_element
