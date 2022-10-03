"""
Class inherited from dcc.Graph. Add one method for updating graph layout.
"""

from dash import dcc
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
            'responsive': True,
            'modeBarButtonsToRemove': ['lasso2d',
                                       'zoomOut2d',
                                       'zoomIn2d',
                                       'resetScale2d',
                                       'hoverClosestCartesian',
                                       'hoverCompareCartesian',
                                       'toggleSpikelines'],
            # 'modeBarStyle': {'orientation': 'v'}, # Deprecated in Dash 1.17.0

            'displaylogo': False,

        }

    def adjust_graph(self,
                     subtitle=None,
                     subset_graph=False,
                     x_ax="",
                     y_ax=""):
        """
        Override graph layout for app use
        """
        new_title = update_title(self.figure.layout.title.text) 
        self.figure.update_layout(
            autosize=True,
            margin=dict(
                l=50,
                r=10,
                b=10,
                t=67,
                pad=0
            ),
            width=None,
            height=None,
            title={
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'text': '<span style="font-size: 1.2vw;">' + new_title + '</span>' 
                # 'text': new_title,
                # 'font': {'size': new_size_font}
                #, 'family': 'verdana'}
            }
        )
        self.figure.update_xaxes(title='<span style="font-size: 1vw;">'  + x_ax + '</span>',
                                 #title='<b>{}</b>'.format(x_ax),
                                 automargin=True,
                                 #tickfont='<span style="size: 1vw"></span>'
                                 # title_font_size=17
                                 )
                                # title_font_family="verdana")
        self.figure.update_yaxes(title='<span style="font-size: 1vw;">' + y_ax + '</span>',
                                 #title='<b>{}</b>'.format(y_ax),
                                 automargin=True,
                                 # title_font_size=17)
                                 #title_font_family="verdana")
                                 )

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
    patt = re.compile('^(.+)<span.+?(Predict: .*|Proba: .*)?</span>$')
    try:
        list_non_empty_str_matches = [x for x in patt.findall(title)[0] if x != '']
        updated = ' - '.join(map(str, list_non_empty_str_matches))
    except:
        updated = title
    return updated
