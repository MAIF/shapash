from typing import Optional, Tuple

import numpy as np
import pandas as pd



def select_data_from_prediction_picking(round_dataframe: pd.DataFrame, selected_data: dict) -> pd.DataFrame:
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