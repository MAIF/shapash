"""
Integration tests for webapp tests
Reference : https://dash.plotly.com/testing
"""
import pytest
import time
import random
from os.path import dirname, abspath, join
import pandas as pd
from category_encoders import OrdinalEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier, LGBMRegressor

from shapash.explainer.smart_explainer import SmartExplainer

pytestmark = pytest.mark.selenium


def get_explainer(case, use_groups=False):
    data_path = dirname(dirname(abspath(__file__)))
    titanic_df = pd.read_pickle(join(data_path, "data", "clean_titanic.pkl"))

    features = ["Pclass", "Age", "Embarked", "Sex"]

    X = titanic_df[features]
    y = titanic_df["Survived"].to_frame()

    onehot = OneHotEncoder(cols=["Pclass"]).fit(X)
    result_1 = onehot.transform(X)
    ordinal = OrdinalEncoder(cols=["Embarked", "Sex"]).fit(result_1)
    titanic_enc = ordinal.transform(result_1)
    encoder = [onehot, ordinal]

    if case == "classification":
        model = LGBMClassifier()

    elif case == "regression":
        y = [random.random() * 100 for _ in range(len(titanic_df))]
        model = LGBMRegressor()

    elif case == "multiclass":
        y = [random.randint(0, 4) for _ in range(len(titanic_df))]
        model = LGBMClassifier()

    X_train, X_test, y_train, y_test = train_test_split(
        titanic_enc,
        y,
        test_size=0.2,
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if use_groups:
        features_groups = {"group1": ["Pclass", "Age"]}
    else:
        features_groups = None

    xpl = SmartExplainer(
        model=model, preprocessing=encoder, title_story="Group features test", features_groups=features_groups
    )
    y_pred = pd.DataFrame(data=y_pred, columns=["pred"], index=X_test.index)

    xpl.compile(X_test, y_pred=y_pred)
    return xpl


def test_classification_dashboard(dash_duo):
    xpl = get_explainer(case="classification")
    xpl.init_app()

    dash_duo.start_server(xpl.smartapp.app)
    dash_duo.wait_for_text_to_equal("h3", "Group features test", timeout=30)
    assert dash_duo.get_logs() == [], "browser console should contain no error"

    time.sleep(3)  # If we don't wait graphs don't have time to display properly

    all_graphs_titles = dash_duo.find_elements(".gtitle")
    expected_graphs_titles = ["Features Importance", "Feature Contribution", "Local Explanation"]
    assert len(all_graphs_titles) == 3, "3 graphs must have titles on the webapp. One graph may not display properly"
    for el in all_graphs_titles:
        assert any([t in el.text for t in expected_graphs_titles]), f"{el.text} is not part of any expected title text"


def test_regression_dashboard(dash_duo):
    xpl = get_explainer(case="regression")
    xpl.init_app()

    dash_duo.start_server(xpl.smartapp.app)
    dash_duo.wait_for_text_to_equal("h3", "Group features test", timeout=30)
    assert dash_duo.get_logs() == [], "browser console should contain no error"

    time.sleep(3)  # If we don't wait graphs don't have time to display properly

    all_graphs_titles = dash_duo.find_elements(".gtitle")
    expected_graphs_titles = ["Features Importance", "Feature Contribution", "Local Explanation"]
    assert len(all_graphs_titles) == 3, "3 graphs must have titles on the webapp. One graph may not display properly"
    for el in all_graphs_titles:
        assert any([t in el.text for t in expected_graphs_titles]), f"{el.text} is not part of any expected title text"


def test_multiclass_dashboard(dash_duo):
    xpl = get_explainer(case="multiclass")
    xpl.init_app()

    dash_duo.start_server(xpl.smartapp.app)
    dash_duo.wait_for_text_to_equal("h3", "Group features test", timeout=30)
    assert dash_duo.get_logs() == [], "browser console should contain no error"

    time.sleep(3)  # If we don't wait graphs don't have time to display properly

    all_graphs_titles = dash_duo.find_elements(".gtitle")
    expected_graphs_titles = ["Features Importance", "Feature Contribution", "Local Explanation"]
    assert len(all_graphs_titles) == 3, "3 graphs must have titles on the webapp. One graph may not display properly"
    for el in all_graphs_titles:
        assert any([t in el.text for t in expected_graphs_titles]), f"{el.text} is not part of any expected title text"


def test_groups_dashboard(dash_duo):
    xpl = get_explainer(case="classification", use_groups=True)
    xpl.init_app()

    dash_duo.start_server(xpl.smartapp.app)
    dash_duo.wait_for_text_to_equal("h3", "Group features test", timeout=30)
    assert dash_duo.get_logs() == [], "browser console should contain no error"

    time.sleep(3)  # If we don't wait graphs don't have time to display properly

    all_graphs_titles = dash_duo.find_elements(".gtitle")
    expected_graphs_titles = ["Features Importance", "Feature Contribution", "Local Explanation"]
    assert len(all_graphs_titles) == 3, "3 graphs must have titles on the webapp. One graph may not display properly"
    for el in all_graphs_titles:
        assert any([t in el.text for t in expected_graphs_titles]), f"{el.text} is not part of any expected title text"
