import plotly.graph_objects as go

from shapash.webapp.utils.MyGraph import MyGraph, split_title_and_subtitle


class TestSplitTitleAndSubtitle:
    def test_no_subtitle(self):
        assert split_title_and_subtitle("Overview") == ("Overview", None)

    def test_sup_subtitle(self):
        assert split_title_and_subtitle("Report<br><sup>Q1 2026</sup>") == ("Report", "Q1 2026")

    def test_span_subtitle_with_attributes(self):
        assert split_title_and_subtitle('Sales<br><span class="sub">Forecast</span>') == ("Sales", "Forecast")

    def test_empty_subtitle_tag(self):
        assert split_title_and_subtitle("Title<br><sup></sup>") == ("Title", "")


class TestMyGraph:
    def _figure(self):
        return go.Figure(layout={"title": {"text": "Report<br><sup>Q1</sup>"}})

    def test_init_special_component_id_config(self):
        graph = MyGraph(self._figure(), component_id="prediction_picking")
        assert graph.config["modeBarButtonsToRemove"] == [
            "zoomOut2d",
            "zoomIn2d",
            "resetScale2d",
            "hoverClosestCartesian",
            "hoverCompareCartesian",
            "toggleSpikelines",
        ]

    def test_init_default_component_id_config(self):
        graph = MyGraph(self._figure(), component_id="other_graph")
        assert "select" in graph.config["modeBarButtonsToRemove"]

    def test_adjust_graph_static_sets_title_and_axes(self):
        figure = self._figure()
        MyGraph.adjust_graph_static(figure, x_ax="X", y_ax="Y")
        assert "Report" in figure.layout.title.text
        assert "Q1" in figure.layout.title.text
        assert figure.layout.xaxis.title.text is not None
        assert figure.layout.yaxis.title.text is not None

    def test_adjust_graph_static_without_subtitle(self):
        figure = go.Figure(layout={"title": {"text": "Plain Title"}})
        MyGraph.adjust_graph_static(figure)
        assert "Plain Title" in figure.layout.title.text
