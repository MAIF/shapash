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
from selenium.webdriver.common.by import By
from selenium.webdriver import ActionChains

from shapash.explainer.smart_explainer import SmartExplainer

pytestmark = pytest.mark.selenium


def get_explainer(case, use_groups=False, use_target=False):
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

    if use_target:
        xpl.compile(X_test, y_pred=y_pred, y_target=y_test)
    else:
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


class TestInteractions:
    xpl = get_explainer(case="classification", use_target=True)
    xpl.init_app()

    def test_settings_dashboard(self, dash_duo):
        dash_duo.start_server(self.xpl.smartapp.app)
        dash_duo.wait_for_text_to_equal("h3", "Group features test", timeout=30)
        assert dash_duo.get_logs() == [], "browser console should contain no error"

        time.sleep(3)  # If we don't wait graphs don't have time to display properly

        dash_duo.find_element("#settings").click()
        dash_duo.wait_for_text_to_equal(".modal-header", "Settings", timeout=30)
        assert dash_duo.get_logs() == [], "browser console should contain no error"

        all_settings = dash_duo.driver.find_elements(By.XPATH, "//div[@id='modal']/div/div[2]/form/div")
        expected_settings_labels = [
            "Number of rows for subset",
            "Number of points for plot",
            "Number of features to plot",
            "Max number of labels for violin plot",
            "Use domain name for features name.",
        ]
        for el in all_settings:
            assert any(
                [t in el.text for t in expected_settings_labels]
            ), f"{el.text} is not part of any expected label settings"

    def test_github_redirection_dashboard(self, dash_duo):
        dash_duo.start_server(self.xpl.smartapp.app)
        dash_duo.wait_for_text_to_equal("h3", "Group features test", timeout=30)
        assert dash_duo.get_logs() == [], "browser console should contain no error"

        time.sleep(3)  # If we don't wait graphs don't have time to display properly

        dash_duo.find_element("#shapash_title").click()
        dash_duo.toggle_window()

        dash_duo.wait_for_element_by_id("repository-container-header", timeout=30)
        assert dash_duo.get_logs() == [], "browser console should contain no error"

    def test_features_importance(self, dash_duo):
        dash_duo.start_server(self.xpl.smartapp.app)
        dash_duo.wait_for_text_to_equal("h3", "Group features test", timeout=30)
        assert dash_duo.get_logs() == [], "browser console should contain no error"

        time.sleep(3)  # If we don't wait graphs don't have time to display properly

        global_importance = dash_duo.find_element("#global_feature_importance")
        feature = global_importance.find_element(By.CSS_SELECTOR, ".ytick").text
        action = ActionChains(dash_duo.driver)
        action.move_to_element(global_importance.find_element(By.CSS_SELECTOR, ".point")).click().perform()

        time.sleep(3)  # Wait for selection to apply

        feature_plot = dash_duo.find_element("#feature_selector")
        feature_plot_title = feature_plot.find_element(By.CSS_SELECTOR, ".gtitle").text

        assert feature in feature_plot_title, "Feature plot should have updated with the selected feature"

    def test_dataset_dashboard(self, dash_duo):
        dash_duo.start_server(self.xpl.smartapp.app)
        dash_duo.wait_for_text_to_equal("h3", "Group features test", timeout=30)
        assert dash_duo.get_logs() == [], "browser console should contain no error"

        time.sleep(3)  # If we don't wait graphs don't have time to display properly

        cell = dash_duo.driver.find_element(By.XPATH, "//table/tbody/tr[2]/td[1]/div")
        cell.click()

        time.sleep(3)  # Wait for selection to apply
        assert dash_duo.get_logs() == [], "browser console should contain no error"

        local_plot = dash_duo.find_element("#detail_feature")
        local_plot_title = local_plot.find_element(By.CSS_SELECTOR, ".gtitle").text

        assert (
            "Local Explanation - Id: " + cell.text in local_plot_title
        ), "Local plot should have updated with the selected index"

    def test_filters_dashboard(self, dash_duo):
        dash_duo.start_server(self.xpl.smartapp.app)
        dash_duo.wait_for_text_to_equal("h3", "Group features test", timeout=30)
        assert dash_duo.get_logs() == [], "browser console should contain no error"

        time.sleep(3)  # If we don't wait graphs don't have time to display properly

        all_tabs = dash_duo.driver.find_elements(By.XPATH, "//ul[@id='tabs']/div/a")
        all_tabs[1].click()

        dash_duo.wait_for_element_by_id("filters", timeout=2)
        all_filters = dash_duo.driver.find_elements(By.XPATH, "//div[@id='filters']/button")
        expected_buttons = ["Add Filter", "Reset all existing filters", "?"]
        for el in all_filters:
            assert any([t in el.text for t in expected_buttons]), f"{el.text} is not part of any expected buttons"

        # Apply a filter to get a subset
        dash_duo.find_element("#add_dropdown_button").click()
        dash_duo.wait_for_element_by_id("react-select-4--value", timeout=30)
        dash_duo.find_element("#react-select-4--value").click()
        dash_duo.wait_for_element_by_css_selector(".Select-menu-outer", timeout=30)
        menu_options = dash_duo.driver.find_elements(By.XPATH, "//div[@class='Select-menu-outer']/div/div/div/div/div")
        index_sex = [opt.text for opt in menu_options].index("Sex")
        menu_options[index_sex].click()
        dash_duo.wait_for_element_by_id("react-select-5--value", timeout=30)
        dash_duo.find_element("#react-select-5--value").click()
        dash_duo.wait_for_element_by_css_selector(".Select-menu-outer", timeout=30)
        menu_options_sex = dash_duo.driver.find_elements(
            By.XPATH, "//div[@class='Select-menu-outer']/div/div/div/div/div"
        )
        menu_options_sex[-1].click()
        dash_duo.find_element("#apply_filter").click()

        time.sleep(3)  # Wait for filters to apply
        assert dash_duo.get_logs() == [], "browser console should contain no error"

        # Apply a second filter to get an empty subset
        dash_duo.find_element("#add_dropdown_button").click()
        dash_duo.wait_for_element_by_id("react-select-6--value", timeout=30)
        dash_duo.find_element("#react-select-6--value").click()
        dash_duo.wait_for_element_by_css_selector(".Select-menu-outer", timeout=30)
        menu_options = dash_duo.driver.find_elements(By.XPATH, "//div[@class='Select-menu-outer']/div/div/div/div/div")
        index_sex = [opt.text for opt in menu_options].index("Sex")
        menu_options[index_sex].click()
        dash_duo.wait_for_element_by_id("react-select-7--value", timeout=30)
        dash_duo.find_element("#react-select-7--value").click()
        dash_duo.wait_for_element_by_css_selector(".Select-menu-outer", timeout=30)
        menu_options_sex = dash_duo.driver.find_elements(
            By.XPATH, "//div[@class='Select-menu-outer']/div/div/div/div/div"
        )
        menu_options_sex[0].click()
        dash_duo.find_element("#apply_filter").click()

        time.sleep(3)  # Wait for filters to apply
        assert dash_duo.get_logs() != [], "browser console should contain an error"

        dash_duo.find_element("#reset_dropdown_button").click()

        time.sleep(3)  # Wait for filters to apply
        assert dash_duo.get_logs() == [], "browser console should contain no error"

    def test_true_vs_predicted_dashboard(self, dash_duo):
        dash_duo.start_server(self.xpl.smartapp.app)
        dash_duo.wait_for_text_to_equal("h3", "Group features test", timeout=30)
        assert dash_duo.get_logs() == [], "browser console should contain no error"

        time.sleep(3)  # If we don't wait graphs don't have time to display properly

        all_tabs = dash_duo.driver.find_elements(By.XPATH, "//ul[@id='tabs']/div/a")
        all_tabs[2].click()

        dash_duo.wait_for_element_by_id("prediction_picking", timeout=30)

        time.sleep(3)  # If we don't wait graphs don't have time to display properly

        graph_picking = dash_duo.find_element("#prediction_picking")
        title = graph_picking.find_element(By.CSS_SELECTOR, ".gtitle")
        assert "True Values Vs Predicted Values" in title.text, f"{title.text} is not the expected title text"

        graph_picking.find_element(By.XPATH, "//div[@class='modebar-group'][2]/a[@data-val='select']").click()

        action = ActionChains(dash_duo.driver)

        # Picking of an empty subset
        grid = graph_picking.find_element(By.CSS_SELECTOR, "rect.nsewdrag")
        action.move_to_element(grid).click_and_hold().move_by_offset(20, -20).release().perform()

        time.sleep(3)  # Wait for picking to apply
        assert dash_duo.get_logs() != [], "browser console should contain an error"

        # Picking of a subset
        plot = graph_picking.find_element(By.CSS_SELECTOR, "g.plot")
        action.move_to_element(plot).click_and_hold().move_by_offset(100, -100).release().perform()

        time.sleep(3)  # Wait for picking to apply
        global_importance = dash_duo.find_element("#global_feature_importance")
        legend = global_importance.find_element(By.CSS_SELECTOR, ".legendtext").text
        assert legend == "Subset", "Features Importance plot should have updated with the Subset legend"
        assert dash_duo.get_logs() == [], "browser console should contain no error"
