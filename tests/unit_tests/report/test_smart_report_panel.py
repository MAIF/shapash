import unittest
from pathlib import Path
import tempfile

import panel as pn
import pandas as pd
import plotly.graph_objects as go

from shapash.report.blocks import ReportBlockMixin, block
from shapash.report.panel_support import apply_report_css


class TestSmartReportPanel(unittest.TestCase):
    def test_panel_plotly_pane_is_responsive(self):
        fig = go.Figure(go.Scatter(x=[1, 2], y=[3, 4]))

        pane = pn.pane.Plotly(fig, config={"responsive": True}, sizing_mode="stretch_width")

        self.assertIsInstance(pane, pn.pane.Plotly)
        self.assertEqual(pane.object, fig)
        self.assertEqual(pane.sizing_mode, "stretch_width")

    def test_report_css_text_loads_stylesheet_content(self):
        css_path = Path(__file__).resolve().parents[3] / "shapash" / "report" / "report_styles.css"
        css = css_path.read_text(encoding="utf-8")

        self.assertIn(".kv-table", css)
        self.assertIn("@media (max-width: 1200px)", css)

    def test_apply_report_css_registers_styles_once(self):
        css_path = Path(__file__).resolve().parents[3] / "shapash" / "report" / "report_styles.css"
        css = css_path.read_text(encoding="utf-8")

        apply_report_css()
        first_count = pn.config.raw_css.count(css)

        apply_report_css()
        second_count = pn.config.raw_css.count(css)

        self.assertEqual(first_count, 1)
        self.assertEqual(second_count, 1)

    def test_apply_report_css_accepts_custom_css_file(self):
        marker_css = ".custom-report-marker{outline:1px solid #f00;}"
        with tempfile.TemporaryDirectory() as tmp_dir:
            custom_css_path = Path(tmp_dir) / "custom.css"
            custom_css_path.write_text(marker_css, encoding="utf-8")

            apply_report_css(custom_css="custom.css", base_dir=tmp_dir)

        self.assertIn(marker_css, pn.config.raw_css)


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
