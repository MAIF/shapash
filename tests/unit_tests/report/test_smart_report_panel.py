import unittest

import panel as pn
import plotly.graph_objects as go

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
