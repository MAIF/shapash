"""
Category_encoder
"""

import pandas as pd
import numpy as np

category_encoder_onehot = "<class 'category_encoders.one_hot.OneHotEncoder'>"
category_encoder_ordinal = "<class 'category_encoders.ordinal.OrdinalEncoder'>"
category_encoder_basen = "<class 'category_encoders.basen.BaseNEncoder'>"
category_encoder_binary = "<class 'category_encoders.binary.BinaryEncoder'>"
category_encoder_targetencoder = "<class 'category_encoders.target_encoder.TargetEncoder'>"

dummies_category_encoder = (category_encoder_onehot,
                            category_encoder_binary,
                            category_encoder_basen)

no_dummies_category_encoder = (category_encoder_ordinal,
                               category_encoder_targetencoder)

supported_category_encoder = (category_encoder_onehot,
                              category_encoder_binary,
                              category_encoder_basen,
                              category_encoder_ordinal,
                              category_encoder_targetencoder)


def inv_transform_ce(x_in, encoding):
    """
    Choose and apply the reversed transformation for the given encoding.

    Parameters
    ----------
    x_in : pandas.DataFrame
        Prediction set.
    encoding : list
        A list of category encoder (OrdinalEncoder/OnehotEncoder/BaseNEncoder/BinaryEncoder/TargetEncoder)
        or a list of dict

    Returns
    -------
    pandas.Dataframe
        The reversed transformation for the given encoding.
    """
    if str(type(encoding)) == category_encoder_ordinal:
        rst = inv_transform_ordinal(x_in, encoding.mapping)

    elif str(type(encoding)) == category_encoder_onehot:
        x = encoding.reverse_dummies(x_in, encoding.mapping)
        rst = inv_transform_ordinal(x, encoding.ordinal_encoder.mapping)

    elif str(type(encoding)) == category_encoder_basen:
        x = reverse_basen(x_in, encoding)
        rst = inv_transform_ordinal(x, encoding.ordinal_encoder.mapping)

    elif str(type(encoding)) == category_encoder_binary:
        x = reverse_basen(x_in, encoding.base_n_encoder)
        rst = inv_transform_ordinal(x, encoding.base_n_encoder.ordinal_encoder.mapping)

    elif str(type(encoding)) == category_encoder_targetencoder:
        rst = inv_transform_target(x_in, encoding)

    elif str(type(encoding)) == "<class 'list'>":
        rst = inv_transform_ordinal(x_in, encoding)

    else:
        raise Exception(f"{encoding.__class__.__name__} not supported, no inverse done.")

    return rst


def inv_transform_target(x_in, enc_target):
    """
    Reversed transformation for target encoded data using target encoded value.

    If a targeted value is linked to multiple label, we raise an exception.

    example of multi label :
    original mapping :
        target   value
           0.5    A
           0.5    B
           0.3    C
    new mapping :
        target value
          0.5   A / B
          0.3   C

    Parameters
    ----------
    x_in : pandas.DataFrame
        Prediction set.
    enc_target : list
        A list containing a TargetEncoder from category encoder.

    Returns
    -------
    pandas.Dataframe
        The reversed dataframe.
    """
    for tgt_enc in enc_target.ordinal_encoder.mapping:
        name_target = tgt_enc.get('col')
        mapping_ordinal = enc_target.mapping[name_target]
        mapping_target = tgt_enc.get('mapping')
        reverse_target = pd.Series(mapping_target.index.values, index=mapping_target)
        rst_target = pd.concat([reverse_target, mapping_ordinal], axis=1, join='inner').fillna(value='NaN')
        aggregate = rst_target.groupby(1)[0].apply(lambda x: ' / '.join(map(str, x)))
        if aggregate.shape[0] != rst_target.shape[0]:
            raise Exception('Multiple label found for the same value in TargetEncoder on col '+str(name_target) +'.')
            #print("Warning in inverse TargetEncoder - col " + str(name_target) + ": Multiple label for the same value, "
            #                                                                   "each label will be separate using : / ")

        transco = {'col': name_target,
                   'mapping': pd.Series(data=aggregate.index, index=aggregate.values),
                   'data_type': 'object'}
        x_in = inv_transform_ordinal(x_in, [transco])
    return x_in


def inv_transform_ordinal(x_in, encoding):
    """
    Reversed transformation based on ordinal category encoder.

    Parameters
    ----------
    x_in : pandas.DataFrame
        Prediction set.
    encoding : list
        A list of dict containing the col, the mapping and the data_type use for reversed transformation.

    Returns
    -------
    pandas.Dataframe
        The reversed dataframe.
    """
    for switch in encoding:
        col_name = switch.get('col')
        if not col_name in x_in.columns:
            raise Exception(f'Columns {col_name} not in dataframe.')
        column_mapping = switch.get('mapping')
        inverse = pd.Series(data=column_mapping.index, index=column_mapping.values)
        x_in[col_name] = x_in[col_name].map(inverse).astype(switch.get('data_type'))
    return x_in


def reverse_basen(x_in, encoding):
    """
    Reversed dummies based on baseN category encoder.

    Parameters
    ----------
    x_in : pandas.DataFrame
        Prediction set.
    encoding : list
        A list of dict containing the col, the mapping and the data_type use for reversed transformation.

    Returns
    -------
    pandas.Dataframe
        The reversed dummies dataframe for a given baseN encoding encoding.
    """
    x = x_in.copy(deep=True)
    out_cols = x.columns.values.tolist()
    for ind_enc in range(len(encoding.mapping)):
        col_list = encoding.mapping[ind_enc].get('mapping').columns.tolist()
        insert_at = out_cols.index(col_list[0])

        if encoding.base == 1:
            value_array = np.array([int(col0.split('_')[-1]) for col0 in col_list])
        else:
            len0 = len(col_list)
            value_array = np.array([encoding.base ** (len0 - 1 - i) for i in range(len0)])
        x.insert(insert_at, encoding.cols[ind_enc], np.dot(x[col_list].values, value_array.T))
        x.drop(col_list, axis=1, inplace=True)
        out_cols = x.columns.values.tolist()
    return x


def calc_inv_contrib_ce(x_contrib, encoding):
    """
    Reversed contribution when category encoder and/or a dict is used.
    If category encoder create multiple columns, we use the mapping to find which contribution columns to sum.
    Else we return the contribution without aggregate.

    Parameters
    ----------
    x_contrib : pandas.DataFrame
        Contributions set.
    encoding : category_encoders, list, dict
        The processing apply to the original data.

    Returns
    -------
    pandas.DataFrame
        The aggregate contributions depending on which processing is apply.
    """
    if str(type(encoding)) in dummies_category_encoder:
        if str(type(encoding)) in category_encoder_binary:
            encoding = encoding.base_n_encoder
        drop_col = []
        for switch in encoding.mapping:
            col_in = switch.get('col')
            mod = switch.get('mapping').columns.tolist()
            insert_at = x_contrib.columns.tolist().index(mod[0])
            x_contrib.insert(insert_at, col_in, x_contrib[mod].sum(axis=1))
            drop_col += mod
        x_contrib.drop(drop_col, axis=1, inplace=True)
        return x_contrib
    else:
        return x_contrib
