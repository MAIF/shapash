"""
shap_backend allow to compute contribution if needed
No tuning possible here:
The goal we are pursuing is to allow the user to have
a first level of explanation with very little code
the idea here is to purpose a simple implementation
to compute a first explanation in one line of code
but if you are looking a particular technique
we invite you to code it yourself.
You can check the reference https://github.com/slundberg/shap
You can also watch the tutorials which shows how to use shapash
with contributions calculated by lime or eli5 library
"""
import pandas as pd
import shap

def shap_contributions(model, x_df):
    """
    Compute the local shapley contributions of each individual,
    feature.
    Using shap to

    Parameters
    ----------
    model: model object from sklearn, catboost, xgboost or lightgbm library
        this model is used to choose a shap explainer and to compute
        shapley values
    x_df: pd.DataFrame


    Returns
    -------
    np.array or list of np.array

    """
    simple_tree_model = (
        "<class 'sklearn.ensemble._forest.ExtraTreesClassifier'>",
        "<class 'sklearn.ensemble._forest.ExtraTreesRegressor'>",
        "<class 'sklearn.ensemble._forest.RandomForestClassifier'>",
        "<class 'sklearn.ensemble._forest.RandomForestRegressor'>",
        "<class 'sklearn.ensemble._gb.GradientBoostingClassifier'>",
        "<class 'sklearn.ensemble._gb.GradientBoostingRegressor'>",
        "<class 'lightgbm.sklearn.LGBMClassifier'>",
        "<class 'lightgbm.sklearn.LGBMRegressor'>",
        "<class 'xgboost.sklearn.XGBClassifier'>",
        "<class 'xgboost.sklearn.XGBRegressor'>"
    )

    catboost_model = (
        "<class 'catboost.core.CatBoostClassifier'>",
        "<class 'catboost.core.CatBoostRegressor'>"
    )

    linear_model = (
        "<class 'sklearn.linear_model._logistic.LogisticRegression'>",
        "<class 'sklearn.linear_model._base.LinearRegression'>"
    )

    svm_model = (
        "<class 'sklearn.svm._classes.SVC'>",
        "<class 'sklearn.svm._classes.SVR'>"
    )

    if str(type(model)) in simple_tree_model:
        explainer = shap.TreeExplainer(model)
        contributions = explainer.shap_values(x_df)
        print("Backend: Shap TreeExplainer")

    elif str(type(model)) in catboost_model:
        explainer = shap.TreeExplainer(model)
        contributions = explainer.shap_values(x_df)
        print("Backend: Shap TreeExplainer")

    elif str(type(model)) in linear_model:
        explainer = shap.LinearExplainer(model,x_df)
        contributions = explainer.shap_values(x_df)
        print("Backend: Shap LinearExplainer")

    elif str(type(model)) in svm_model:
        explainer = shap.KernelExplainer(model.predict, x_df)
        contributions = explainer.shap_values(x_df)
        print("Backend: Shap KernelExplainer")

    else:
        raise ValueError(
            """
            model not supported by shapash, please compute contributions
            by yourself before using shapash 
            """
        )

    return contributions