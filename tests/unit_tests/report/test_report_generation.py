import unittest
from pathlib import Path
from unittest.mock import patch
from typing import Any, cast

import numpy as np
import panel as pn
import pandas as pd
import plotly.graph_objects as go

from shapash.backend import BaseBackend
from shapash.explainer import SmartExplainer
from shapash.report.blocks import ReportBlockMixin, block
from shapash.report.core import _inject_favicon_links, _resolve_report_title, build_navigation_bar
from shapash.report.panel_support import apply_report_css

import pytest

def dummy_metric(y_true, y_pred):
    return 0.75


class _DummyModel:
    def __init__(self):
        self.alpha = 0.1
        self.depth = 4
        self.classes_ = np.array([0, 1])

    def predict(self, x):
        return np.zeros(len(x))

    def predict_proba(self, x):
        return np.array([[0.1, 0.9], [0.8, 0.2], [0.2, 0.8]])


class _DummyBackend(BaseBackend):
    name = "dummy"

    def __init__(self, model, preprocessing=None, masker=None):
        super().__init__(model=model, preprocessing=preprocessing)
        self.masker = masker

    def run_explainer(self, x: pd.DataFrame) -> dict:
        return {"contributions": self._build_contributions(x)}

    @staticmethod
    def _build_contributions(x: pd.DataFrame) -> np.ndarray:
        base = np.array(
            [
                [0.4, 0.1],
                [0.2, 0.3],
                [0.5, 0.2],
            ]
        )
        return np.dstack((-base, base))


class _DummyPlot:
    def __init__(self):
        self._style_dict = {
            "dummy": "style",
            "report_feature_distribution": {"train": "#f4c000", "test": "#2255aa"},
        }

    def _tuning_round_digit(self):
        return None

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


class _TestSmartExplainer(SmartExplainer):
    def __init__(self):
        model = _DummyModel()
        super().__init__(
            model=model, backend=_DummyBackend(model),
            features_dict={"age": "Age", "income": "Income"},
            colors_dict={"report_feature_distribution": {"train": "#f4c000", "test": "#2255aa"}, "default": "#2255aa"},
        )
        self.plot = _DummyPlot()
        self.explainer.plot = self.plot
        self.explainer.get_interaction_values = self.get_interaction_values

    def get_interaction_values(self, selection=None):
        return np.array([[[0.0, 0.5], [0.5, 0.0]], [[0.0, 0.4], [0.4, 0.0]]])


def _build_runtime() -> ReportBlockMixin:
    x_train = pd.DataFrame({"age": [20, 30, 40], "income": [100, 200, 150]})
    x_test = pd.DataFrame({"age": [21, 31, 41], "income": [110, 210, 160]})
    y_train = pd.Series([0, 1, 1], name="target")
    y_test = pd.Series([1, 0, 1], name="target")
    explainer = _TestSmartExplainer()
    explainer.compile(
        x=x_test,
        y_pred=pd.Series([1, 0, 1], index=x_test.index),
        y_target=y_test,
    )
    return ReportBlockMixin(
        explainer=explainer,
        x_train=x_train,
        y_train=y_train,
        y_test=y_test,
        max_points=10,
    )


def test_report_runtime_uses_prediction_data_from_underlying_explainer():
    runtime = _build_runtime()

    assert runtime.x_init is runtime.explainer.x_init
    assert runtime.df_train_test["data_train_test"].value_counts().to_dict() == {"test": 3, "train": 3}


def test_inject_favicon_links_replaces_existing_icons():
        html_text = """
<html>
    <head>
        <title>Panel</title>
        <link rel=\"icon\" href=\"https://cdn.example/icon.png\">
    </head>
    <body></body>
</html>
"""
        updated = _inject_favicon_links(html_text, favicon_href="data:image/png;base64,AAA")

        assert "<title>Panel</title>" in updated
        assert "https://cdn.example/icon.png" not in updated
        assert '<link rel="icon" href="data:image/png;base64,AAA">' in updated
        assert '<link rel="shortcut icon" href="data:image/png;base64,AAA">' in updated
        assert '<link rel="apple-touch-icon" href="data:image/png;base64,AAA">' in updated


def test_resolve_report_title_falls_back_to_default_when_missing():
        class _RuntimeWithoutTitle:
                smart_explainer = type("SE", (), {"title_story": ""})()

        assert _resolve_report_title(_RuntimeWithoutTitle(), report_title=None) == "Shapash Report"


class TestSmartReportPanel(unittest.TestCase):

    def test_report_css_text_loads_stylesheet_content(self):
        css_path = Path(__file__).resolve().parents[3] / "shapash" / "report" / "assets" / "report_styles.css"
        css = css_path.read_text(encoding="utf-8")

        self.assertIn(".kv-table", css)
        self.assertIn("@media (max-width: 1200px)", css)
        main_rules = css.split(".main-report", maxsplit=1)[1].split("}", maxsplit=1)[0]
        base_rules = css.split(".report-sidebar", maxsplit=1)[1].split("}", maxsplit=1)[0]
        responsive_rules = css.split("@media (max-width: 1200px)", maxsplit=1)[1]
        sidebar_rules = responsive_rules.split(".report-sidebar", maxsplit=1)[1].split("}", maxsplit=1)[0]
        self.assertIn("overflow: visible !important;", main_rules)
        self.assertIn("position: sticky;", base_rules)
        self.assertIn("position: static;", sidebar_rules)

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
        return [pn.widgets.Select(label="Feature", options=["a", "b"], value="a")]

    @block
    def block_plotly_allowed(self, title: str = "Plotly"):
        fig = go.Figure(go.Scatter(x=[1, 2], y=[3, 4]))
        return [pn.pane.Plotly(fig)]

    @block
    def block_bind_allowed(self, title: str = "Bind"):
        selector = pn.widgets.Select(label="Feature", options=["a", "b"], value="a")
        selected_panel = pn.panel(pn.bind(cast(Any, lambda selected: pn.pane.Markdown(selected)), selector))
        return [selector, selected_panel]

    @block
    def block_panel_type_not_allowed(self, title: str = "Button"):
        return [pn.widgets.Button(name="Click")]

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
        self.assertIsInstance(result.objects[2], pn.pane.DataFrame)
        self.assertFalse(result.objects[2].index)

        result_with_index = runtime.block_model_analysis(show_index=True)
        self.assertTrue(result_with_index.objects[2].index)

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

    def test_class_explainability_uses_selected_binary_or_all_multiclass_labels(self):
        binary_runtime = _build_runtime()
        with patch.object(
            binary_runtime.explainer.plot,
            "features_importance",
            wraps=binary_runtime.explainer.plot.features_importance,
        ) as binary_importance, patch.object(
            binary_runtime.explainer.plot,
            "contribution_plot",
            wraps=binary_runtime.explainer.plot.contribution_plot,
        ) as binary_contributions:
            binary_runtime.block_class_explainability()

        self.assertEqual([call.kwargs["label"] for call in binary_importance.call_args_list], [1])
        self.assertEqual([call.kwargs["label"] for call in binary_contributions.call_args_list], [1, 1])

        multiclass_runtime = _build_runtime()
        multiclass_runtime.explainer._classes = [0, 1, 2]
        with patch.object(
            multiclass_runtime.explainer,
            "check_label_name",
            side_effect=lambda class_code, origin=None: ([0, 1, 2].index(class_code), class_code, f"Class {class_code}"),
        ), patch.object(
            multiclass_runtime.explainer.plot,
            "features_importance",
            wraps=multiclass_runtime.explainer.plot.features_importance,
        ) as multiclass_importance, patch.object(
            multiclass_runtime.explainer.plot,
            "contribution_plot",
            wraps=multiclass_runtime.explainer.plot.contribution_plot,
        ) as multiclass_contributions:
            result = multiclass_runtime.block_class_explainability()

        self.assertIsInstance(result, pn.Column)
        self.assertEqual([call.kwargs["label"] for call in multiclass_importance.call_args_list], [0, 1, 2])
        self.assertEqual([call.kwargs["label"] for call in multiclass_contributions.call_args_list], [0, 0, 1, 1, 2, 2])
        class_links = multiclass_runtime.class_navigation_items["class-explainability"]
        self.assertEqual([item["label"] for item in class_links], ["Class 0", "Class 1", "Class 2"])
        anchors = [
            child
            for item in result.objects
            if isinstance(item, pn.Column)
            for child in item.objects
            if isinstance(child, pn.pane.HTML)
        ]
        self.assertEqual([anchor.object for anchor in anchors], [
            f'<div id="{item["anchor"]}" class="scroll-anchor"></div>' for item in class_links
        ])

        nav = build_navigation_bar(
            [
                {
                    "type": "group",
                    "params": {"title": "Model explainability"},
                    "_section_id": "model-explainability",
                    "blocks": [
                        {
                            "type": "class_explainability",
                            "params": {"title": "Explained classes"},
                            "_section_id": "class-explainability",
                        }
                    ],
                }
            ],
            {"class-explainability": class_links},
        )
        for class_link in class_links:
            self.assertIn(f'href="#{class_link["anchor"]}"', nav.object)
            self.assertIn(class_link["label"], nav.object)

    def test_class_explainability_can_include_class_specific_interactions(self):
        runtime = _build_runtime()
        runtime.explainer._classes = [0, 1, 2]

        with patch.object(
            runtime.explainer,
            "check_label_name",
            side_effect=lambda class_code, origin=None: ([0, 1, 2].index(class_code), class_code, f"Class {class_code}"),
        ), patch.object(
            runtime.explainer.plot,
            "top_interactions_plot",
            wraps=runtime.explainer.plot.top_interactions_plot,
        ) as interactions_plot:
            result = runtime.block_class_explainability(include_interactions=True, nb_top_interactions=3)

        self.assertIsInstance(result, pn.Column)
        self.assertEqual([call.kwargs["label"] for call in interactions_plot.call_args_list], [0, 1, 2])
        self.assertEqual([call.kwargs["nb_top_interactions"] for call in interactions_plot.call_args_list], [3, 3, 3])
        interaction_panes = [
            pane
            for section in result.objects
            if isinstance(section, pn.Column)
            for pane in section.select(pn.pane.Plotly)
            if pane.object.data[0].type == "scatter"
        ]
        self.assertEqual(len(interaction_panes), 3)

    def test_block_contribution_plot_single_and_all_features(self):
        runtime = _build_runtime()

        single_result = runtime.block_contribution_plot(feature="age", title=None)
        all_result = runtime.block_contribution_plot(include_all_features=True, title="All")

        self.assertIsInstance(single_result, pn.Column)
        self.assertIn("Age", single_result.objects[0].object)
        self.assertIsInstance(single_result.objects[1], pn.pane.Plotly)
        self.assertIsInstance(all_result.objects[1], pn.widgets.Select)
        self.assertEqual(type(all_result.objects[2]).__name__, "ParamFunction")

    def test_block_top_interactions_plot_renders_plotly(self):
        runtime = _build_runtime()

        result = runtime.block_top_interactions_plot(title="Top interactions", nb_top_interaction=3)

        self.assertIsInstance(result, pn.Column)
        self.assertIn("Top interactions", result.objects[0].object)
        self.assertIsInstance(result.objects[1], pn.pane.Plotly)

    def test_block_top_interactions_plot_passes_label_to_plotter(self):
        runtime = _build_runtime()

        with patch.object(runtime.explainer.plot, "top_interactions_plot", wraps=runtime.explainer.plot.top_interactions_plot) as mocked_top:
            runtime.block_top_interactions_plot(title="Top interactions", nb_top_interaction=3, class_label="class_1")

        self.assertEqual(mocked_top.call_args.kwargs["label"], "class_1")

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
        target_stats = analysis_result.objects[2].objects[0]
        self.assertIsInstance(target_stats, pn.pane.DataFrame)
        self.assertTrue(target_stats.index)

        analysis_without_index = runtime.block_target_analysis(title="Target", show_index=False)
        self.assertFalse(analysis_without_index.objects[2].objects[0].index)

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

    def test_smart_explainer_required(self):
        rbm = ReportBlockMixin()
        with pytest.raises(ValueError):
            rbm._require_smart_explainer("block_type")
