"""
Unit test smart predictor
"""

import unittest
from shapash.explainer.smart_explainer import SmartPredictor
from shapash.explainer.smart_explainer import SmartExplainer
import pandas as pd
import numpy as np
import catboost as cb
from catboost import Pool
import category_encoders as ce


class TestSmartPredictor(unittest.TestCase):
    """
    Unit test Smart Predictor class
    """
    def test_init_1(self):
        """
        Test init smart predictor
        """
        df = pd.DataFrame(range(0, 5), columns=['id'])
        df['y'] = df['id'].apply(lambda x: 1 if x < 2 else 0)
        df['x1'] = np.random.randint(1, 123, df.shape[0])
        df['x2'] = ["S", "M", "S", "D", "M"]
        df = df.set_index('id')
        encoder = ce.OrdinalEncoder(cols=["x2"], handle_unknown="None")
        encoder_fitted = encoder.fit(df)
        df_encoded = encoder_fitted.transform(df)
        clf = cb.CatBoostClassifier(n_estimators=1).fit(df_encoded[['x1', 'x2']], df_encoded['y'])

        columns_dict = {0:"x1",2:"x2"}
        label_dict = {0:"Yes",1:"No"}

        postprocessing = {"x2": {
            "type": "transcoding",
            "rule": {"S": "single", "M": "married", "D": "divorced"}}}
        features_dict={"x1": "age", "x2": "family_situation"}

        features_types = {features: str(df[features].dtypes) for features in df.columns}

        predictor_1 = SmartPredictor(features_dict, clf,
                 columns_dict, features_types, label_dict,
                 encoder_fitted,postprocessing)

        mask_params = {
            'features_to_hide': None,
            'threshold': None,
            'positive': True,
            'max_contrib': 1
        }

        predictor_2 = SmartPredictor(features_dict, clf,
                                     columns_dict, features_types, label_dict,
                                     encoder_fitted, postprocessing,
                                     mask_params)

        assert hasattr(predictor_1, 'model')
        assert hasattr(predictor_1, 'features_dict')
        assert hasattr(predictor_1, 'label_dict')
        assert hasattr(predictor_1, '_case')
        assert hasattr(predictor_1, '_classes')
        assert hasattr(predictor_1, 'columns_dict')
        assert hasattr(predictor_1, 'features_types')
        assert hasattr(predictor_1, 'preprocessing')
        assert hasattr(predictor_1, 'postprocessing')
        assert hasattr(predictor_1, 'mask_params')
        assert hasattr(predictor_2, 'mask_params')

        assert predictor_1.model == clf
        assert predictor_1.features_dict == features_dict
        assert predictor_1.label_dict == label_dict
        assert predictor_1._case == "classification"
        assert predictor_1._classes == [0,1]
        assert predictor_1.columns_dict == columns_dict
        assert predictor_1.preprocessing == encoder_fitted
        assert predictor_1.postprocessing == postprocessing

        assert predictor_2.mask_params == mask_params

    def add_input_1(self):
        """
        Test add_input method from smart predictor
        """
        df = pd.DataFrame(range(0, 5), columns=['id'])
        df['y'] = df['id'].apply(lambda x: 1 if x < 2 else 0)
        df['x1'] = np.random.randint(1, 123, df.shape[0])
        df['x2'] = ["S", "M", "S", "D", "M"]
        df = df.set_index('id')
        encoder = ce.OrdinalEncoder(cols=["x2"], handle_unknown="None")
        encoder_fitted = encoder.fit(df)
        df_encoded = encoder_fitted.transform(df)
        clf = cb.CatBoostClassifier(n_estimators=1).fit(df_encoded[['x1', 'x2']], df_encoded['y'])
        columns_dict = {0: "x1", 2: "x2"}
        label_dict = {0: "Yes", 1: "No"}
        postprocessing = {"x2": {
            "type": "transcoding",
            "rule": {"S": "single", "M": "married", "D": "divorced"}}}
        features_dict = {"x1": "age", "x2": "family_situation"}
        features_types = {features: str(df[features].dtypes) for features in df.columns}

        ypred = pd.DataFrame({"y": [1, 0], "id": [0, 1]}).set_index("id")
        shap_values = clf.get_feature_importance(Pool(df_encoded), type="ShapValues")

        xpl = SmartExplainer(features_dict, label_dict)
        xpl.compile(contributions=shap_values[:, :-1], x=df_encoded, model=clf, preprocessing=encoder_fitted)

        predictor_1 = SmartPredictor(features_dict, clf,
                                     columns_dict, features_types, label_dict,
                                     encoder_fitted, postprocessing)

        predictor_1.add_input(x = df[["x1","x2"]], contributions = shap_values[:,:-1])
        xpl_contrib = xpl.contributions
        predictor_1_contrib = predictor_1.contributions

        assert hasattr(predictor_1, "x")
        assert hasattr(predictor_1, "x_preprocessed")
        assert not hasattr(predictor_1, "ypred")
        assert hasattr(predictor_1, "contributions")
        assert predictor_1.x.shape == predictor_1.x_preprocessed.shape
        assert all(feature in predictor_1.x.columns for feature in predictor_1.x_preprocessed.columns)

        assert isinstance(predictor_1_contrib, list)
        assert len(predictor_1_contrib) == len(xpl_contrib)
        for i, contrib in enumerate(predictor_1_contrib):
            pd.testing.assert_frame_equal(contrib, xpl_contrib[i])
            assert all(contrib.index == xpl_contrib[i].index)
            assert all(contrib.columns == xpl_contrib[i].columns)
            assert all(contrib.dtypes == xpl_contrib[i].dtypes)

        predictor_1.add_input(ypred=ypred)
        assert hasattr(predictor_1, "ypred")
        assert predictor_1.ypred.shape[0] == predictor_1.x.shape[0]
        assert all(predictor_1.ypred.index == predictor_1.x.index)












