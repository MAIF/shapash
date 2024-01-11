import numpy as np
import pandas as pd
import pytest

from shapash.utils.model import predict_error

y1 = pd.DataFrame(data=np.array([1, 2, 3]), columns=["pred"])
expected1 = pd.DataFrame(data=np.array([0.0, 0.0, 0.0]), columns=["_error_"])

y2 = pd.DataFrame(data=np.array([0, 2, 3]), columns=["pred"])
expected2 = pd.DataFrame(data=np.array([1, 0, 0]), columns=["_error_"])


@pytest.mark.parametrize(
    "y_target, y_pred, case, expected",
    [
        (None, None, "classification", None),
        (y1, y1, "classification", None),
        (y1, None, "regression", None),
        (None, y1, "regression", None),
        (y1, y1, "regression", expected1),
        (y2, y1, "regression", expected2),
    ],
)
def test_predict_error_works(y_target, y_pred, case, expected):
    result = predict_error(y_target, y_pred, case)
    if result is not None:
        assert not pd.testing.assert_frame_equal(result, expected)
    else:
        assert result == expected
