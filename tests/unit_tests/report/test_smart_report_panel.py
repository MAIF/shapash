import unittest

import panel as pn
import pandas as pd
import plotly.graph_objects as go

from shapash.report.blocks import ReportBlockMixin, block
from shapash.report.panel_support import apply_report_css, make_plotly_pane, report_css_text


class TestSmartReportPanel(unittest.TestCase):
    def test_make_plotly_pane_returns_panel_plotly(self):
        fig = go.Figure(go.Scatter(x=[1, 2], y=[3, 4]))

        pane = make_plotly_pane(fig)

        self.assertIsInstance(pane, pn.pane.Plotly)
        self.assertEqual(pane.object, fig)
        self.assertEqual(pane.sizing_mode, "stretch_width")

    def test_report_css_text_loads_stylesheet_content(self):
        css = report_css_text()

        self.assertIn(".kv-table", css)
        self.assertIn("@media (max-width: 1200px)", css)

    def test_apply_report_css_registers_styles_once(self):
        css = report_css_text()

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
