import unittest

import plotly.graph_objects as go

from shapash.report.smart_report.layout import build_html_page
from shapash.report.smart_report.panel_support import panel_resource_tags, render_plotly_pane_html


class TestSmartReportPanel(unittest.TestCase):
    def test_panel_resource_tags_include_panel_dependencies(self):
        tags = panel_resource_tags()

        self.assertIn("cdn.holoviz.org/panel", tags)
        self.assertIn("panel.min.js", tags)
        self.assertIn("cdn.bokeh.org", tags)
        self.assertIn("plotly", tags)

    def test_render_plotly_pane_html_returns_panel_fragment(self):
        fig = go.Figure(go.Scatter(x=[1, 2], y=[3, 4]))

        html = render_plotly_pane_html(fig)

        self.assertIn('class="panel-plot"', html)
        self.assertIn("data-root-id=", html)
        self.assertIn("panel.models.plotly.PlotlyPlot", html)

    def test_build_html_page_includes_panel_resources(self):
        html = build_html_page(body="<div>Body</div>")

        self.assertIn("cdn.holoviz.org/panel", html)
        self.assertIn("panel.min.js", html)
        self.assertIn("cdn.bokeh.org", html)
        self.assertNotIn("cdn.plot.ly", html)
