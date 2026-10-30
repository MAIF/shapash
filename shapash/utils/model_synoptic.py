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
    "<class 'sklearn.ensemble._hist_gradient_boosting.gradient_boosting.HistGradientBoostingClassifier'>",
    "<class 'sklearn.ensemble._hist_gradient_boosting.gradient_boosting.HistGradientBoostingRegressor'>",
    "<class 'sklearn.tree._classes.DecisionTreeClassifier'>",
    "<class 'sklearn.tree._classes.DecisionTreeRegressor'>",
    "<class 'sklearn.tree._classes.ExtraTreeClassifier'>",
    "<class 'sklearn.tree._classes.ExtraTreeRegressor'>",
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
    "<class 'sklearn.linear_model._logistic.LogisticRegressionCV'>",
    "<class 'sklearn.linear_model._ridge.Ridge'>",
    "<class 'sklearn.linear_model._ridge.RidgeCV'>",
    "<class 'sklearn.linear_model._coordinate_descent.Lasso'>",
    "<class 'sklearn.linear_model._coordinate_descent.LassoCV'>",
    "<class 'sklearn.linear_model._coordinate_descent.ElasticNet'>",
    "<class 'sklearn.linear_model._coordinate_descent.ElasticNetCV'>",
    "<class 'sklearn.linear_model._bayes.BayesianRidge'>",
    "<class 'sklearn.linear_model._bayes.ARDRegression'>",
    "<class 'sklearn.linear_model._huber.HuberRegressor'>",
    "<class 'sklearn.linear_model._theil_sen.TheilSenRegressor'>",
    "<class 'sklearn.linear_model._quantile.QuantileRegressor'>",
    "<class 'sklearn.linear_model._stochastic_gradient.SGDRegressor'>",
    "<class 'sklearn.linear_model._passive_aggressive.PassiveAggressiveRegressor'>",
    "<class 'sklearn.linear_model._least_angle.Lars'>",
    "<class 'sklearn.linear_model._least_angle.LarsCV'>",
    "<class 'sklearn.linear_model._least_angle.LassoLars'>",
    "<class 'sklearn.linear_model._least_angle.LassoLarsCV'>",
    "<class 'sklearn.linear_model._least_angle.LassoLarsIC'>",
    "<class 'sklearn.linear_model._omp.OrthogonalMatchingPursuit'>",
    "<class 'sklearn.linear_model._omp.OrthogonalMatchingPursuitCV'>",
)

svm_model = (
    "<class 'sklearn.svm._classes.SVC'>",
    "<class 'sklearn.svm._classes.SVR'>",
    "<class 'sklearn.svm._classes.LinearSVC'>",
    "<class 'sklearn.svm._classes.LinearSVR'>",
)

simple_tree_model = simple_tree_model_sklearn + xgboost_model + lightgbm_model

dict_model_feature = {
    **{model: ["length"] for model in simple_tree_model_sklearn + linear_model + svm_model},
    "<class 'lightgbm.sklearn.LGBMClassifier'>": ["booster_", "feature_name"],
    "<class 'lightgbm.sklearn.LGBMRegressor'>": ["booster_", "feature_name"],
    "<class 'lightgbm.basic.Booster'>": ["feature_names"],
    "<class 'xgboost.sklearn.XGBClassifier'>": ["get_booster", "feature_names"],
    "<class 'xgboost.sklearn.XGBRegressor'>": ["get_booster", "feature_names"],
    "<class 'xgboost.core.Booster'>": ["feature_names"],
    "<class 'catboost.core.CatBoostClassifier'>": ["feature_names_"],
    "<class 'catboost.core.CatBoostRegressor'>": ["feature_names_"],
}
