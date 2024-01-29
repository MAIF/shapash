"""
Model Synoptic Module
"""

simple_tree_model_sklearn = (
    "<class 'sklearn.ensemble._forest.ExtraTreesClassifier'>",
    "<class 'sklearn.ensemble._forest.ExtraTreesRegressor'>",
    "<class 'sklearn.ensemble._forest.RandomForestClassifier'>",
    "<class 'sklearn.ensemble._forest.RandomForestRegressor'>",
    "<class 'sklearn.ensemble._gb.GradientBoostingClassifier'>",
    "<class 'sklearn.ensemble._gb.GradientBoostingRegressor'>",
)
xgboost_model = (
    "<class 'xgboost.sklearn.XGBClassifier'>",
    "<class 'xgboost.sklearn.XGBRegressor'>",
    "<class 'xgboost.core.Booster'>",
)

lightgbm_model = (
    "<class 'lightgbm.sklearn.LGBMClassifier'>",
    "<class 'lightgbm.sklearn.LGBMRegressor'>",
    "<class 'lightgbm.basic.Booster'>",
)

catboost_model = ("<class 'catboost.core.CatBoostClassifier'>", "<class 'catboost.core.CatBoostRegressor'>")

linear_model = (
    "<class 'sklearn.linear_model._logistic.LogisticRegression'>",
    "<class 'sklearn.linear_model._base.LinearRegression'>",
)

svm_model = ("<class 'sklearn.svm._classes.SVC'>", "<class 'sklearn.svm._classes.SVR'>")

simple_tree_model = simple_tree_model_sklearn + xgboost_model + lightgbm_model

dict_model_feature = {
    "<class 'sklearn.ensemble._forest.ExtraTreesClassifier'>": ["length"],
    "<class 'sklearn.ensemble._forest.ExtraTreesRegressor'>": ["length"],
    "<class 'sklearn.ensemble._forest.RandomForestClassifier'>": ["length"],
    "<class 'sklearn.ensemble._forest.RandomForestRegressor'>": ["length"],
    "<class 'sklearn.ensemble._gb.GradientBoostingClassifier'>": ["length"],
    "<class 'sklearn.ensemble._gb.GradientBoostingRegressor'>": ["length"],
    "<class 'sklearn.linear_model._logistic.LogisticRegression'>": ["length"],
    "<class 'sklearn.linear_model._base.LinearRegression'>": ["length"],
    "<class 'sklearn.svm._classes.SVC'>": ["length"],
    "<class 'sklearn.svm._classes.SVR'>": ["length"],
    "<class 'lightgbm.sklearn.LGBMClassifier'>": ["booster_", "feature_name"],
    "<class 'lightgbm.sklearn.LGBMRegressor'>": ["booster_", "feature_name"],
    "<class 'lightgbm.basic.Booster'>": ["feature_names"],
    "<class 'xgboost.sklearn.XGBClassifier'>": ["get_booster", "feature_names"],
    "<class 'xgboost.sklearn.XGBRegressor'>": ["get_booster", "feature_names"],
    "<class 'xgboost.core.Booster'>": ["feature_names"],
    "<class 'catboost.core.CatBoostClassifier'>": ["feature_names_"],
    "<class 'catboost.core.CatBoostRegressor'>": ["feature_names_"],
}
