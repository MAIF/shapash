import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import panel as pn
import pandas as pd
import plotly.graph_objects as go

from shapash.report.blocks import ReportBlockMixin, block
from shapash.report.panel_support import apply_report_css


def dummy_metric(y_true, y_pred):
    return 0.75


class _DummyModel:
    def __init__(self):
        self.alpha = 0.1
        self.depth = 4

    def predict(self, x):
        return np.zeros(len(x))


class _DummyPlot:
    def __init__(self):
        self._style_dict = {"dummy": "style"}

    def correlations_plot(self, *args, **kwargs):
        return go.Figure(go.Scatter(x=[1, 2], y=[2, 1]))

    def features_importance(self, *args, **kwargs):
        return go.Figure(go.Bar(x=["age", "income"], y=[0.7, 0.3]))

    def contribution_plot(self, *args, **kwargs):
        return go.Figure(go.Bar(x=["age"], y=[1.0]))

    def interactions_plot(self, *args, **kwargs):
        return go.Figure(go.Scatter(x=[1, 2], y=[3, 4]))

    def top_interactions_plot(self, *args, **kwargs):
        return go.Figure(go.Scatter(x=[2, 3], y=[4, 5]))

    def _select_indices_interactions_plot(self, selection=None, max_points=200):
        return [0, 1], None


class _DummyExplainer:
    def __init__(self, x_init):
        self.x_init = x_init
        self.x_encoded = x_init
        self.y_pred = [1, 0, 1]
        self.model = _DummyModel()
        self.preprocessing = None
        self.postprocessing = None
        self.features_dict = {"age": "Age", "income": "Income"}
        self.inv_features_dict = {"Age": "age", "Income": "income"}
        self.colors_dict = {
            "report_feature_distribution": {"train": "#f4c000", "test": "#2255aa"},
            "default": "#2255aa",
        }
        self.plot = _DummyPlot()
        self.columns_dict = {0: "age", 1: "income"}
        self._case = "classification"
        self.y_target = [1, 0, 1]
        self.proba_values = None

    def get_interaction_values(self, selection=None):
        return np.array([[0.0, 0.5], [0.5, 0.0]])

    def check_label_name(self, label):
        return 1, "class_1", "class_1"

    def predict_proba(self):
        self.proba_values = np.array([[0.1, 0.9], [0.8, 0.2], [0.2, 0.8]])


def _build_runtime() -> ReportBlockMixin:
    x_train = pd.DataFrame({"age": [20, 30, 40], "income": [100, 200, 150]})
    x_test = pd.DataFrame({"age": [21, 31, 41], "income": [110, 210, 160]})
    y_train = pd.Series([0, 1, 1], name="target")
    y_test = pd.Series([1, 0, 1], name="target")
    explainer = _DummyExplainer(x_test)
    return ReportBlockMixin(explainer=explainer, x_train=x_train, y_train=y_train, y_test=y_test, max_points=10)


class TestSmartReportPanel(unittest.TestCase):

    def test_report_css_text_loads_stylesheet_content(self):
        css_path = Path(__file__).resolve().parents[3] / "shapash" / "report" / "assets" / "report_styles.css"
        css = css_path.read_text(encoding="utf-8")

        self.assertIn(".kv-table", css)
        self.assertIn("@media (max-width: 1200px)", css)

    def test_apply_report_css_registers_styles_once(self):
        css_path = Path(__file__).resolve().parents[3] / "shapash" / "report" / "assets"  / "report_styles.css"
        css = css_path.read_text(encoding="utf-8")

        apply_report_css()
        first_count = pn.config.raw_css.count(css)

        apply_report_css()
        second_count = pn.config.raw_css.count(css)

        self.assertEqual(first_count, 1)
        self.assertEqual(second_count, 1)


class _DummyBlocks(ReportBlockMixin):
    @block
    def block_demo(self, title: str = "Demo"):
        return [pn.pane.Markdown("Body")]

    @block
    def block_dynamic_title(self, title: str = ""):
        return "Resolved title", [pn.pane.Markdown("Dynamic body")]

    @block
    def block_scalar_body(self, title: str = "Scalar"):
        return "plain text"

    @block
    def block_table(self, title: str = "Table"):
        return [pn.pane.DataFrame(pd.DataFrame({"a": [1], "b": [2]}))]

    @block
    def block_badge_row(self, title: str = "Badges"):
        return [pn.Row(pn.pane.Markdown("One"), pn.pane.Markdown("Two"))]

    @block
    def block_select_allowed(self, title: str = "Selector"):
        return [pn.widgets.Select(name="Feature", options=["a", "b"], value="a")]

    @block
    def block_plotly_allowed(self, title: str = "Plotly"):
        fig = go.Figure(go.Scatter(x=[1, 2], y=[3, 4]))
        return [pn.pane.Plotly(fig)]

    @block
    def block_bind_allowed(self, title: str = "Bind"):
        selector = pn.widgets.Select(name="Feature", options=["a", "b"], value="a")
        selected_panel = pn.panel(pn.bind(lambda selected: pn.pane.Markdown(selected), selector))
        return [selector, selected_panel]

    @block
    def block_panel_type_not_allowed(self, title: str = "HTML"):
        return [pn.pane.HTML("<b>html</b>")]

    @block
    def block_non_panel_type_not_allowed(self, title: str = "Object"):
        return [object()]


class TestBlockDecorator(unittest.TestCase):
    def test_block_decorator_wraps_with_title_from_signature(self):
        runtime = _DummyBlocks()

        result = runtime.block_demo()

        self.assertIsInstance(result, pn.Column)
        self.assertEqual(len(result.objects), 2)
        self.assertIsInstance(result.objects[0], pn.pane.Markdown)
        self.assertIn("Demo", result.objects[0].object)

    def test_block_decorator_supports_dynamic_title_tuple(self):
        runtime = _DummyBlocks()

        result = runtime.block_dynamic_title()

        self.assertIsInstance(result, pn.Column)
        self.assertEqual(len(result.objects), 2)
        self.assertIsInstance(result.objects[0], pn.pane.Markdown)
        self.assertIn("Resolved title", result.objects[0].object)

    def test_block_decorator_coerces_scalar_body_to_markdown(self):
        runtime = _DummyBlocks()

        result = runtime.block_scalar_body()

        self.assertIsInstance(result, pn.Column)
        self.assertEqual(len(result.objects), 2)
        self.assertIsInstance(result.objects[1], pn.pane.Markdown)
        self.assertIn("plain text", result.objects[1].object)

    def test_block_decorator_auto_stylizes_body_by_type(self):
        runtime = _DummyBlocks()

        text_result = runtime.block_demo()
        table_result = runtime.block_table()

        self.assertIn("content-block", text_result.objects[1].css_classes)
        self.assertIn("kv-table", table_result.objects[1].css_classes)

    def test_block_decorator_auto_styles_badge_rows(self):
        runtime = _DummyBlocks()

        result = runtime.block_badge_row()

        badge_row = result.objects[1]
        self.assertIsInstance(badge_row, pn.Row)
        self.assertIn("badge-pill", badge_row.objects[0].css_classes)
        self.assertIn("badge-pill", badge_row.objects[1].css_classes)

    def test_block_decorator_allows_select_and_plotly(self):
        runtime = _DummyBlocks()

        select_result = runtime.block_select_allowed()
        plotly_result = runtime.block_plotly_allowed()

        self.assertIsInstance(select_result.objects[1], pn.widgets.Select)
        self.assertIsInstance(plotly_result.objects[1], pn.pane.Plotly)

    def test_block_decorator_allows_bind_param_function(self):
        runtime = _DummyBlocks()

        result = runtime.block_bind_allowed()

        self.assertIsInstance(result.objects[1], pn.widgets.Select)
        self.assertEqual(type(result.objects[2]).__name__, "ParamFunction")

    def test_block_decorator_rejects_panel_type_without_style_definition(self):
        runtime = _DummyBlocks()

        with self.assertRaises(TypeError) as context:
            runtime.block_panel_type_not_allowed()

        self.assertIn("Unsupported Panel object type returned", str(context.exception))
        self.assertIn("Allowed Panel return types", str(context.exception))

    def test_block_decorator_rejects_non_panel_return_type(self):
        runtime = _DummyBlocks()

        with self.assertRaises(TypeError) as context:
            runtime.block_non_panel_type_not_allowed()

        self.assertIn("Unsupported block return type", str(context.exception))


class TestReportBlockMixinBuiltins(unittest.TestCase):
    def test_block_text_accepts_dict_content(self):
        runtime = _build_runtime()

        result = runtime.block_text(title="Info", content={"project": "shapash", "version": "1.0"})

        self.assertIsInstance(result, pn.Column)
        self.assertIn("Info", result.objects[0].object)
        self.assertIn("**project**", result.objects[1].object)

    def test_block_global_analysis_renders_stats_table(self):
        runtime = _build_runtime()
        fake_stats = {"Rows": 3, "Columns": 2}

        with patch("shapash.report.blocks.perform_global_dataframe_analysis", return_value=fake_stats), patch(
            "shapash.report.blocks.stats_to_table", return_value=pd.DataFrame({"Prediction dataset": [3]})
        ):
            result = runtime.block_global_analysis(title="Global")

        self.assertIsInstance(result, pn.Column)
        self.assertIn("Global", result.objects[0].object)
        self.assertIsInstance(result.objects[1], pn.pane.DataFrame)

    def test_block_model_analysis_renders_metadata(self):
        runtime = _build_runtime()

        with patch("shapash.report.blocks.importlib.metadata.version", return_value="9.9.9"):
            result = runtime.block_model_analysis()

        self.assertIsInstance(result, pn.Column)
        self.assertIn("Model information", result.objects[0].object)
        self.assertIn("**Model used**", result.objects[1].object)

    def test_block_performance_metrics_builds_badges(self):
        runtime = _build_runtime()

        result = runtime.block_performance_metrics(
            title="Perf", metrics=[{"path": f"{__name__}.dummy_metric", "name": "Dummy metric"}]
        )

        self.assertIsInstance(result, pn.Column)
        self.assertIn("Perf", result.objects[0].object)
        row = result.objects[1]
        self.assertIsInstance(row, pn.Row)
        self.assertIn("Dummy metric", row.objects[0].object)

    def test_block_feature_distribution_uses_feature_label_when_title_is_none(self):
        runtime = _build_runtime()

        with patch("shapash.report.blocks.plot_distribution", return_value=go.Figure(go.Scatter(x=[1], y=[1]))):
            result = runtime.block_feature_distribution(feature="age", title=None)

        self.assertIsInstance(result, pn.Column)
        self.assertIn("Age", result.objects[0].object)
        self.assertIsInstance(result.objects[1], pn.pane.Plotly)

    def test_block_correlations_and_feature_importance_return_plotly_panes(self):
        runtime = _build_runtime()

        corr_result = runtime.block_correlations_plot(title="Corr")
        fi_result = runtime.block_feature_importance(title="FI")

        self.assertIsInstance(corr_result.objects[1], pn.pane.Plotly)
        self.assertIsInstance(fi_result.objects[1], pn.pane.Plotly)

    def test_block_contribution_plot_single_and_all_features(self):
        runtime = _build_runtime()

        single_result = runtime.block_contribution_plot(feature="age", title=None)
        all_result = runtime.block_contribution_plot(include_all_features=True, title="All")

        self.assertIsInstance(single_result, pn.Column)
        self.assertIn("Age", single_result.objects[0].object)
        self.assertIsInstance(single_result.objects[1], pn.pane.Plotly)
        self.assertIsInstance(all_result.objects[1], pn.widgets.Select)
        self.assertEqual(type(all_result.objects[2]).__name__, "ParamFunction")

    def test_block_interactions_plot_default_pair_uses_resolved_labels(self):
        runtime = _build_runtime()

        with patch(
            "shapash.report.blocks.compute_sorted_variables_interactions_list_indices", return_value=[(0, 1)]
        ):
            result = runtime.block_interactions_plot(title=None)

        self.assertIsInstance(result, pn.Column)
        self.assertIn("Age / Income", result.objects[0].object)
        self.assertIsInstance(result.objects[1], pn.pane.Plotly)

    def test_block_top_interactions_plot_renders_plotly(self):
        runtime = _build_runtime()

        result = runtime.block_top_interactions_plot(title="Top interactions", nb_top_interaction=3)

        self.assertIsInstance(result, pn.Column)
        self.assertIn("Top interactions", result.objects[0].object)
        self.assertIsInstance(result.objects[1], pn.pane.Plotly)

    def test_block_target_distribution_and_analysis_render(self):
        runtime = _build_runtime()
        fake_fig = go.Figure(go.Scatter(x=[1], y=[1]))
        fake_univariate = {"target": {"count": 3, "na_count": 0}}

        with patch("shapash.report.blocks.plot_distribution", return_value=fake_fig), patch(
            "shapash.report.blocks.compute_col_types", return_value={"target": "numeric"}
        ), patch("shapash.report.blocks.perform_univariate_dataframe_analysis", return_value=fake_univariate):
            dist_result = runtime.block_target_distribution(title=None)
            analysis_result = runtime.block_target_analysis(title="Target")

        self.assertIsInstance(dist_result.objects[1], pn.pane.Plotly)
        self.assertIsInstance(analysis_result, pn.Column)
        self.assertIn("Target", analysis_result.objects[0].object)
        self.assertIsInstance(analysis_result.objects[2], pn.Row)

    def test_block_confusion_lift_and_univariate_render(self):
        runtime = _build_runtime()
        fake_fig = go.Figure(go.Scatter(x=[0, 1], y=[1, 0]))
        fake_univariate = {
            "age": {"count": 3, "na_count": 0},
            "income": {"count": 3, "na_count": 0},
            "data_train_test": {"count": 6},
        }

        with patch("shapash.report.blocks.plot_confusion_matrix", return_value=fake_fig), patch(
            "shapash.report.blocks.plot_lift_curve", return_value=fake_fig
        ), patch("shapash.report.blocks.compute_col_types", return_value={"age": "numeric", "income": "numeric"}), patch(
            "shapash.report.blocks.perform_univariate_dataframe_analysis", return_value=fake_univariate
        ), patch("shapash.report.blocks.plot_distribution", return_value=fake_fig):
            confusion_result = runtime.block_confusion_matrix(title="CM")
            lift_result = runtime.block_lift_curve(title="Lift")
            univariate_result = runtime.block_univariate_analysis()

        self.assertIsInstance(confusion_result.objects[1], pn.pane.Plotly)
        self.assertIsInstance(lift_result.objects[1], pn.pane.Plotly)
        self.assertIsNotNone(runtime.explainer.proba_values)
        self.assertIsInstance(univariate_result.objects[1], pn.widgets.Select)
        self.assertEqual(type(univariate_result.objects[2]).__name__, "ParamFunction")
