"""
Unit test smart predictor
"""

import unittest
from shapash.explainer.smart_explainer import SmartPredictor
import pandas as pd
import numpy as np
import catboost as cb
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

        predictor_1 = SmartPredictor(features_dict, clf,
                 columns_dict, label_dict,
                 encoder_fitted,postprocessing)

        mask_params = {
            'features_to_hide': None,
            'threshold': None,
            'positive': True,
            'max_contrib': 1
        }

        predictor_2 = SmartPredictor(features_dict, clf,
                                     columns_dict, label_dict,
                                     encoder_fitted, postprocessing,
                                     mask_params)

        assert hasattr(predictor_1, 'model')
        assert hasattr(predictor_1, 'features_dict')
        assert hasattr(predictor_1, 'label_dict')
        assert hasattr(predictor_1, '_case')
        assert hasattr(predictor_1, '_classes')
        assert hasattr(predictor_1, 'columns_dict')
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



