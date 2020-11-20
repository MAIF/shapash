"""
Modele Synoptic Module
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
