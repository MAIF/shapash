import re

from dash import dcc


class MyGraph(dcc.Graph):
    """Class inherited from dcc.Graph. Add one method for updating graph layout."""

    def __init__(self, figure, id, style=None, **kwds):
        super().__init__(**kwds)
        self.figure = figure
        self.style = style if style is not None else {}
        self.id = id
        # self.config = {'modeBarButtons': {'pan2d': True}}
        if id == "prediction_picking":
            self.config = {
                # Graph is responsive
                "responsive": True,
                "modeBarButtonsToRemove": [
                    "zoomOut2d",
                    "zoomIn2d",
                    "resetScale2d",
                    "hoverClosestCartesian",
                    "hoverCompareCartesian",
                    "toggleSpikelines",
                ],
                "displaylogo": False,
            }
            # 'modeBarStyle': {'orientation': 'v'}, # Deprecated in Dash 1.17.0
        else:
            self.config = {
                # Graph is responsive
                "responsive": True,
                # Graph don't have select box button
                "modeBarButtonsToRemove": [
                    "lasso2d",
                    "zoomOut2d",
                    "zoomIn2d",
                    "resetScale2d",
                    "hoverClosestCartesian",
                    "hoverCompareCartesian",
                    "toggleSpikelines",
                    "select",
                ],
                "displaylogo": False,
            }

    @staticmethod
    def adjust_graph_static(figure, x_ax="", y_ax=""):
        """
        Override graph layout for app use
        ----------------------------------------
        x_ax: title of the x-axis
        y_ax: title of the y-axis
        ---------------------------------------
        """
        new_title = update_title(figure.layout.title.text)
        figure.update_layout(
            autosize=True,
            margin=dict(l=50, r=10, b=10, t=67, pad=0),
            width=None,
            height=None,
            title={
                "y": 0.95,
                "x": 0.5,
                "xanchor": "center",
                "yanchor": "top",
                # update title and font-size of the title
                "text": '<span style="font-size: 1.2vw;">' + new_title + "</span>",
            },
        )
        # update x title and font-size of the title
        figure.update_xaxes(title='<span style="font-size: 1vw;">' + x_ax + "</span>", automargin=True)
        # update y title and font-size of the title
        figure.update_yaxes(title='<span style="font-size: 1vw;">' + y_ax + "</span>", automargin=True)


def update_title(title):
    """
    adapt title content the app layout
    Parameters
    ----------
    title : str
        string to ba adapted
    Returns
    -------
    str
    """
    patt = re.compile("^(.+)<span.+?(Predict: .*|Proba: .*)?</span>$")
    try:
        list_non_empty_str_matches = [x for x in patt.findall(title)[0] if x != ""]
        updated = " - ".join(map(str, list_non_empty_str_matches))
    except Exception:
        updated = title
    return updated
