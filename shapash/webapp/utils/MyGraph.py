import re

from dash import dcc


class MyGraph(dcc.Graph):
    """Class inherited from dcc.Graph. Add one method for updating graph layout."""

    def __init__(self, figure, id, style=None, **kwds):
        super().__init__(
            id=id,
            figure=figure,
            style=style if style is not None else {},
            **kwds,
        )

        self.figure = figure

        if id in ["prediction_picking", "clusters"]:
            self.config = {
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
        else:
            self.config = {
                "responsive": True,
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
        main_title, subtitle = split_title_and_subtitle(figure.layout.title.text)
        title_html = f'<span style="font-size: calc(1.5vh + 0.5vw);">{main_title}</span>'
        if subtitle:
            title_html += (
                '<br><span style="display:block; font-size:calc(1vh + 0.4vw);margin-top:1vh;">' + subtitle + "</span>"
            )
        figure.update_layout(
            autosize=True,
            margin=dict(l=50, r=10, b=10, t=67, pad=0),
            width=None,
            height=None,
            title={
                "y": 0.94,
                "x": 0.5,
                "xanchor": "center",
                "yanchor": "top",
                "text": title_html,
            },
        )
        # update x title and font-size of the title
        figure.update_xaxes(
            title='<span style="font-size: calc(0.45rem + 0.7vw);">' + x_ax + "</span>", automargin=True
        )
        # update y title and font-size of the title
        figure.update_yaxes(
            title='<span style="font-size: calc(0.45rem + 0.7vw);">' + y_ax + "</span>", automargin=True
        )


def split_title_and_subtitle(title: str):
    """
    Split an HTML-formatted title into a main title and an optional subtitle.

    The function searches for a specific separator in the input string:
    a line break followed by either a <sup>...</sup> or <span ...>...</span> block,
    i.e., "<br><sup>...</sup>" or "<br><span ...>...</span>". If the pattern is
    present and the entire string matches this structure, it returns a tuple with
    the text before the separator as the main title and the inner text of the
    <sup> or <span> element as the subtitle. If no such pattern is found, it
    returns the original title and None.

    Parameters
    ----------
    title : str
        The full title string, optionally containing "<br><sup>...</sup>" or
        "<br><span ...>...</span>" at the end.

    Returns
    -------
    tuple[str, Optional[str]]
        (main_title, subtitle) where subtitle is None when no subtitle is detected.
        If a subtitle tag is present but empty, an empty string is returned.

    Notes
    -----
    - The match is anchored to the start and end of the string; the subtitle (if any)
    must appear as the final part of the title.
    - Attributes inside the <span> tag are allowed and ignored.
    - Whitespace is preserved as-is.

    Examples
    --------
    >>> split_title_and_subtitle("Report<br><sup>Q1 2026</sup>")
    ('Report', 'Q1 2026')
    >>> split_title_and_subtitle('Sales<br><span class="sub">Forecast</span>')
    ('Sales', 'Forecast')
    >>> split_title_and_subtitle("Overview")
    ('Overview', None)
    """
    match = re.match(r"^(.*?)(?:<br><(?:sup|span)[^>]*>(.*?)</(?:sup|span)>)?$", title)
    if match:
        main_title = match.group(1)
        subtitle = match.group(2) if match.lastindex and match.lastindex >= 2 else None
        return main_title, subtitle
    return title, None
