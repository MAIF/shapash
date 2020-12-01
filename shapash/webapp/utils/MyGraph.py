"""
Class inherited from dcc.Graph. Add one method for updating graph layout.
"""

import dash_core_components as dcc
from math import floor
import re


class MyGraph(dcc.Graph):

    def __init__(self, figure, id, style={}, **kwds):
        super().__init__(**kwds)
        self.figure = figure
        self.style = style
        self.id = id
        # self.config = {'modeBarButtons': {'pan2d': True}}
        self.config = {
            'modeBarButtonsToRemove': ['select2d', 'lasso2d', 'zoomOut2d', 'zoomIn2d', 'resetScale2d',
                                       'hoverClosestCartesian', 'hoverCompareCartesian', 'toggleSpikelines'],
            #'modeBarStyle': {'orientation': 'v'}, # Deprecated in Dash 1.17.0
            'displaylogo': False,

        }

    def adjust_graph(self, subset_graph=False, title_size_adjust=False):
        """
        Override graph layout for app use
        """
        new_title = update_title(self.figure.layout.title.text) + (" <b>< Subset ></b>" if subset_graph else "")
        if title_size_adjust:
            new_size_font = floor(self.figure.layout.title.font.size * min(60 / len(new_title), 1))
        else:
            new_size_font = self.figure.layout.title.font.size
        self.figure.update_layout(
            autosize=True,
            margin=dict(
                l=50,
                r=10,
                b=0,
                t=50,
                pad=0
            ),
            width=None,
            height=None,
            title={
                'y': 1,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'text': new_title,
                'font': {'size': new_size_font}
            }
        )
        self.figure.update_xaxes(title='', automargin=True)
        self.figure.update_yaxes(title='', automargin=True)


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
    patt = re.compile('^(.+)<span.+?( - Proba: .*)?</span>$')
    try:
        updated = ''.join(map(str, patt.findall(title)[0]))
    except:
        updated = title
    return updated
