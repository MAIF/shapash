import numpy as np
import pandas as pd
import pytest

from shapash.utils.model import predict_error

y1 = pd.DataFrame(data=np.array([1, 2, 3]), columns=["pred"])
y2 = pd.DataFrame(data=np.array([0, 2, 3]), columns=["pred"])
y3 = pd.DataFrame(data=np.array([2, 2, 3]), columns=["pred"])

expected1 = pd.DataFrame(data=np.array([0.0, 0.0, 0.0]), columns=["_error_"])
expected2 = pd.DataFrame(data=np.array([1, 0, 0]), columns=["_error_"])
expected3 = pd.DataFrame(data=np.array([0, 0, 0]), columns=["_error_"])
expected_proba1 = pd.DataFrame({"_error_": [0.9, 0.6, 0.4]})
expected_proba2 = pd.DataFrame({"_error_": [0.5, 0.4, 0.6]})

proba_values1 = pd.DataFrame(
        [[0.1, 0.7, 0.2],
         [0.3, 0.4, 0.3],
         [0.2, 0.2, 0.6]],
        columns=[1, 2, 3]
    )

proba_values2 = pd.DataFrame(
        [
            [0.2, 0.5, 0.3],
            [0.1, 0.6, 0.3],
            [0.3, 0.3, 0.4],
        ],
        columns=[1, 2, 3]
    )

classes = [1, 2, 3]

@pytest.mark.parametrize(
    "y_target, y_pred, model_type, proba_values, classes, expected",
    [
        # -------------------------------
        # Classification — invalid inputs
        # -------------------------------
        (None, None, "classification", None, None, None),
        (y1, None, "classification", None, None, None),
        (None, y1, "classification", None, None, None),

        # -------------------------------
        # Classification — simple 0/1 error
        # -------------------------------
        (y1, y1, "classification", None, None, expected3),
        (y2, y1, "classification", None, None, expected2),

        # -------------------------------
        # Classification — with proba
        # error = |1 - P(true_class)|
        # -------------------------------
        (y1, y1, "classification", proba_values1, classes, expected_proba1),
        (y3, y1, "classification", proba_values2, classes, expected_proba2),

        # -------------------------------
        # Regression — invalid inputs
        # -------------------------------
        (y1, None, "regression", None, None, None),
        (None, y1, "regression", None, None, None),

        # -------------------------------
        # Regression — working cases
        # -------------------------------
        (y1, y1, "regression", None, None, expected1),
        (y2, y1, "regression", None, None, expected2),
    ],
)
def test_predict_error_works(y_target, y_pred, model_type, proba_values, classes, expected):
    result = predict_error(y_target, y_pred, model_type, proba_values, classes)

    if expected is None:
        assert result is None
    else:
        # DataFrame comparison
        pd.testing.assert_frame_equal(result, expected)
