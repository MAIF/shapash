"""
Transform Module
"""
from shapash.utils.inverse_category_encoder import inv_transform_ce
from shapash.utils.inverse_columntransformer import inv_transform_ct
from shapash.utils.inverse_category_encoder import supported_category_encoder
from shapash.utils.inverse_columntransformer import columntransformer

# TODO
# encode targeted variable ? from sklearn.preprocessing import LabelEncoder
# make an easy version for dict, not writing all mapping

def inverse_transform(x_pred, preprocessing=None):
    """
    Reverse transformation giving a preprocessing.

    Preprocessing could be :
        - a single category_encoders
        - a single ColumnTransformer
        - list with multiple category_encoders with optional (dict, list of dict)
        - list with a single ColumnTransformer with optional (dict, list of dict)
        - dict
        - list of dict

    Preprocessing return an error when using ColumnTransformer and category encoding in the same preprocessing,
    the category encoding must be used in the ColumnTransformer.

    If ColumnTransformer is used, there is an inverse transformation for each transformers,
    so a column could be multi-processed, that's why inverse colnames are prefixed by the transformers names.

    Parameters
    ----------
    x_pred : pandas.DataFrame
        Prediction set.
    preprocessing : category_encoders, ColumnTransformer, list, dict, optional (default: None)
        The processing apply to the original data

    Returns
    -------
    pandas.Dataframe
        return the dataframe before preprocessing.
    """

    if preprocessing is None:
        return x_pred
    else:
        # Transform preprocessing into a list
        list_encoding = preprocessing_tolist(preprocessing)

        # Check encoding are supported
        use_ct, use_ce = check_supported_inverse(list_encoding)

        # Apply Inverse Transform
        x_inverse = x_pred.copy()

        for encoding in list_encoding:
            if use_ct:
                x_inverse = inv_transform_ct(x_inverse, encoding)
            else:
                x_inverse = inv_transform_ce(x_inverse, encoding)
        return x_inverse

def preprocessing_tolist(preprocess):
    """
    Transform preprocess into a list, if preprocess contains a dict, transform the dict into a list of dict.

    Parameters
    ----------
    preprocess : category_encoders, ColumnTransformer, list, dict, optional (default: None)
        The processing apply to the original data

    Returns
    -------
    List
        A list containing all preprocessing.
    """
    list_encoding = preprocess if isinstance(preprocess, list) else [preprocess]
    list_encoding = [[x] if isinstance(x, dict) else x for x in list_encoding]
    return list_encoding

def check_supported_inverse(list_encoding):
    """
    Check that all transformation are supported.
        - a single category encoders transformer
        - a single columns transformers
        - list of multiple category encoder with optional (dict, list of dict)
        - list with a single transformer with optional (dict, list of dict)
        - dict
        - list of dict

    If a dict is used, the dict must contain, a col, a mapping and a data_type.
    Example :
        input_dict1['col'] = 'my_col'
        input_dict1['mapping'] = pd.Series(data=['C', 'D', np.nan], index=['C', 'D', 'missing'])
        input_dict1['data_type'] = 'object'

    Parameters
    ----------
    list_encoding : list
        A list containing at least one transformation

    Returns
    -------
        use_ct : boolean
            true if column transformer is used
        use_ce : boolean
            true if category encoder is used

    """

    use_ct = False
    use_ce = False

    for enc in list_encoding:
        if str(type(enc)) in columntransformer:
            use_ct = True

        elif str(type(enc)) in supported_category_encoder:
            use_ce = True

        # Check we have a list of dict
        elif isinstance(enc, list):
            for enc_dict in enc:
                if isinstance(enc_dict, dict):
                    # Check dict structure : col - mapping - data_type
                    if not all(struct in enc_dict for struct in ('col', 'mapping', 'data_type')):
                        raise Exception(f'{enc_dict} should have col, mapping and data_type as keys.')
                else:
                    raise Exception(f'{enc} is not a list of dict.')
        else:
            raise Exception(f'{enc} is not supported yet.')

    # check that encoding don't use ColumnTransformer and Category encoding at the same time
    if use_ct and use_ce:
        raise Exception(
            f"Can't support ColumnTransformer and Category encoding at the same time. "
            f"Use Category encoding in ColumnTransformer")

    # check that Category encoding is apply on different columns
    col = []
    for enc in list_encoding:
        if not str(type(enc)) in ("<class 'list'>",
                                  "<class 'dict'>",
                                  columntransformer):
            col += enc.cols
    duplicate = set([x for x in col if col.count(x) > 1])
    if duplicate:
        raise Exception('Columns ' + str(duplicate) + ' is used in multiple category encoding')

    return use_ct, use_ce


