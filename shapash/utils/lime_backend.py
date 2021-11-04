try:
    from lime import lime_tabular
    is_lime_available = True
except ImportError:
    is_lime_available = False

import pandas as pd
import logging


def lime_contributions(model,
                       x_init,
                       x_train=None,
                       mode="classification",
                       num_classes=None):
    """
        Compute local contribution with the Lime library
        Parameters
        ----------
        model: prediction function. For classifiers, this should be a
                function that takes a numpy array and outputs prediction
                probabilities. For regressors, this takes a numpy array and
                returns the predictions. For ScikitClassifiers, this is
                `classifier.predict_proba()`. For ScikitRegressors, this
                is `regressor.predict()`. The prediction function needs to work
                on multiple feature vectors (the vectors randomly perturbed
                from the data_row).
        x_init  : pd.DataFrame
            preprocessed dataset used by the model to perform the prediction.
        x_train : pd.DataFrame
            Training dataset used as background.
        mode : "classification" or "regression"
        num_classes: int (default :None)
            Number of classes if len(classes)>2
        Returns
        -------
        np.array or list of np.array

    """
    if is_lime_available is False:
        raise ValueError(
            """
            Active Shapley values requires the LIME package,
            which can be installed using 'pip install lime'
            """
                    )

    if x_train is not None:
        x_lime_explainer = x_train
    else:
        logging.warning("No train set passed. We recommend to pass the x_train parameter "
                        "in order to avoid errors.")
        x_lime_explainer = x_init

    explainer = lime_tabular.LimeTabularExplainer(x_lime_explainer.values,
                                                       feature_names=x_lime_explainer.columns,
                                                       mode=mode)
    print("Backend: LIME")
    lime_contrib = []

    for i in x_init.index:

        if mode == "classification" and num_classes <= 2:

            exp = explainer.explain_instance(x_init.loc[i], model.predict_proba)
            lime_contrib.append(dict([[transform_name(b[0], x_init), b[1]] for b in exp.as_list()]))

        elif mode == "classification" and num_classes > 2:

            contribution = []
            for j in range(num_classes):
                list_contrib = []
                df_contrib = pd.DataFrame()
                for i in x_init.index:
                    exp = explainer.explain_instance(
                        x_init.loc[i], model.predict_proba, top_labels=num_classes)
                    list_contrib.append(
                        dict([[transform_name(b[0], x_init), b[1]] for b in exp.as_list(j)]))
                    df_contrib = pd.DataFrame(list_contrib)
                    df_contrib = df_contrib[list(x_init.columns)]
                contribution.append(df_contrib.values)
            return contribution

        else:

            exp = explainer.explain_instance(x_init.loc[i], model.predict)
            lime_contrib.append(dict([[transform_name(b[0], x_init), b[1]] for b in exp.as_list()]))

    contribution = pd.DataFrame(lime_contrib, index=x_init.index)
    contribution = contribution[list(x_init.columns)]

    return contribution


def transform_name(var_name, x_df):
    """Function for transform name of LIME contribution shape to a comprehensive name 

    Args:
        a (str): variable name to transform
        x_df (pd.DataFrame): dataframe with valid name

    Returns:
        str: valid name
    """
    for colname in list(x_df.columns):
        if str(colname) in str(var_name):
            col_rename = colname
    return col_rename
