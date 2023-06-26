from typing import Optional

import numpy as np
import pandas as pd



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
    for p in selected_data['points']:
        row_ids.append(p['customdata'])
    df = round_dataframe.loc[row_ids]
    
    return df


def select_data_from_filters(
        round_dataframe: pd.DataFrame,
        id_feature: list, 
        id_str_modality: list, 
        id_bool_modality: list, 
        id_lower_modality: list, 
        id_date: list, 
        val_feature: list, 
        val_str_modality: list,
        val_bool_modality: list,
        val_lower_modality: list,
        val_upper_modality: list,
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
    id_str_modality : list
        string features ids
    id_bool_modality : list
        boolean features ids
    id_lower_modality : list
        numeric features ids
    id_date : list
        date features ids
    val_feature : list
        features names
    val_str_modality : list
        string modalities selected
    val_bool_modality : list
        boolean modalities selected
    val_lower_modality : list
        lower values of numeric filter
    val_upper_modality : list
        upper values of numeric filter
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
    feature_id = [id_feature[i]['index'] for i in range(len(id_feature))]
    str_id = [id_str_modality[i]['index'] for i in range(len(id_str_modality))]
    bool_id = [id_bool_modality[i]['index'] for i in range(len(id_bool_modality))]
    lower_id = [id_lower_modality[i]['index'] for i in range(len(id_lower_modality))]
    date_id = [id_date[i]['index'] for i in range(len(id_date))]
    df = round_dataframe
    # If there is some filters
    if len(feature_id) > 0:
        for i in range(len(feature_id)):
            # String filter
            if feature_id[i] in str_id:
                position = np.where(np.array(str_id) == feature_id[i])[0][0]
                if ((position is not None) & (val_str_modality[position] is not None)):
                    df = df[df[val_feature[i]].isin(val_str_modality[position])]
            # Boolean filter
            elif feature_id[i] in bool_id:
                position = np.where(np.array(bool_id) == feature_id[i])[0][0]
                if ((position is not None) & (val_bool_modality[position] is not None)):
                    df = df[df[val_feature[i]] == val_bool_modality[position]]
            # Date filter
            elif feature_id[i] in date_id:
                position = np.where(np.array(date_id) == feature_id[i])[0][0]
                if((position is not None) &
                    (start_date[position] < end_date[position])):
                    df = df[((df[val_feature[i]] >= start_date[position]) &
                                (df[val_feature[i]] <= end_date[position]))]
            # Numeric filter
            elif feature_id[i] in lower_id:
                position = np.where(np.array(lower_id) == feature_id[i])[0][0]
                if((position is not None) & (val_lower_modality[position] is not None) &
                    (val_upper_modality[position] is not None)):
                    if (val_lower_modality[position] < val_upper_modality[position]):
                        df = df[(df[val_feature[i]] >= val_lower_modality[position]) &
                                (df[val_feature[i]] <= val_upper_modality[position])]
    
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
    selected_feature = click_data['points'][0]['label'].replace('<b>', '').replace('</b>', '')
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
    list_sub_features = [f for group_features in features_groups.values()
        for f in group_features]
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
    group_name = selected_feature if (
        features_groups is not None and selected_feature in features_groups.keys()
    ) else None
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
    indexes = [d['_index_'] for d in data]
    if list_index is not None and (len(indexes)==len(list_index) or len(indexes)==0):
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
    point = click_data['points'][0]
    point['curveNumber'] = 0
    click_data = {'points':[point]}
    return click_data


def get_figure_zoom(click_zoom: int) -> bool :
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
