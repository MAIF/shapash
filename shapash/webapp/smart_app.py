"""
Main class of Web application Shapash
"""
import dash
from dash import dash_table
import dash_daq as daq
from dash import no_update
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html
from dash import MATCH, ALL
from dash.dependencies import Output, Input, State
from flask import Flask
import pandas as pd
import plotly.graph_objs as go
import random
import numpy as np
import datetime
import re
from math import log10
from shapash.webapp.utils.utils import check_row, round_to_k
from shapash.webapp.utils.MyGraph import MyGraph
from shapash.utils.utils import truncate_str
from shapash.webapp.utils.explanations import Explanations


def create_input_modal(id, label, tooltip):
    return dbc.Row(
        [
            dbc.Label(label, id=f'{id}_label', html_for=id, width=8),
            dbc.Col(
                dbc.Input(id=id, type="number", value=0),
                width=4),
            dbc.Tooltip(tooltip, target=f'{id}_label', placement='bottom'),
        ], className="g-3"
    )


class SmartApp:
    """
        Bridge pattern decoupling the application part from SmartExplainer and SmartPlotter.
        Attributes
        ----------
        explainer: object
            SmartExplainer instance to point to.
    """

    def __init__(self, explainer, settings: dict = None):
        """
        Init on class instantiation, everything to be able to run the app on server.
        Parameters
        ----------
        explainer : SmartExplainer
            SmartExplainer object
        settings : dict
            A dict describing the default webapp settings values to be used
            Possible settings (dict keys) are 'rows', 'points', 'violin', 'features'
            Values should be positive ints
        """
        # APP
        self.server = Flask(__name__)
        self.app = dash.Dash(
            server=self.server,
            external_stylesheets=[dbc.themes.BOOTSTRAP],
        )
        self.app.title = 'Shapash Monitor'
        if explainer.title_story:
            self.app.title += ' - ' + explainer.title_story
        self.explainer = explainer

        # SETTINGS
        self.logo = self.app.get_asset_url('shapash-fond-fonce.png')
        self.color = explainer.plot._style_dict["webapp_button"]
        self.bkg_color = explainer.plot._style_dict["webapp_bkg"]
        self.title_menu_color = explainer.plot._style_dict["webapp_title"]
        self.settings_ini = {
            'rows': 1000,
            'points': 1000,
            'violin': 10,
            'features': 20,
        }
        if settings is not None:
            for k, v in self.settings_ini.items():
                self.settings_ini[k] = settings[k] if k in settings and isinstance(
                    settings[k], int) and 0 < settings[k] else v
        self.settings = self.settings_ini.copy()

        self.predict_col = ['_predict_']
        self.explainer.features_imp = self.explainer.state.compute_features_import(
            self.explainer.contributions)
        if self.explainer._case == 'classification':
            self.label = self.explainer.check_label_name(len(self.explainer._classes) - 1, 'num')[1]
            self.selected_feature = self.explainer.features_imp[-1].idxmax()
            self.max_threshold = int(max([x.applymap(lambda x: round_to_k(x, k=1)).max().max()
                                          for x in self.explainer.contributions]))
        else:
            self.label = None
            self.selected_feature = self.explainer.features_imp.idxmax()
            self.max_threshold = int(self.explainer.contributions.applymap(
                lambda x: round_to_k(x, k=1)).max().max())
        self.list_index = []
        self.subset = None
        self.last_click_data = None

        # DATA
        self.explanations = Explanations()  # To get explanations of "?" buttons
        self.dataframe = pd.DataFrame()
        self.round_dataframe = pd.DataFrame()
        self.init_data()

        # COMPONENTS
        self.components = {
            'menu': {},
            'table': {},
            'graph': {},
            'filter': {},
            'settings': {}
        }
        self.init_components()

        # LAYOUT
        self.skeleton = {
            'navbar': {},
            'body': {}
        }
        self.make_skeleton()
        self.app.layout = html.Div([self.skeleton['navbar'], self.skeleton['body']])

        # CALLBACK
        self.callback_fullscreen_buttons()
        self.init_callback_settings()
        self.callback_generator()

    def init_data(self):
        """
        Method which initializes data from explainer object
        """
        if hasattr(self.explainer, 'y_pred'):
            self.dataframe = self.explainer.x_init.copy()
            if isinstance(self.explainer.y_pred, (pd.Series, pd.DataFrame)):
                self.predict_col = self.explainer.y_pred.columns.to_list()[0]
                self.dataframe = self.dataframe.join(self.explainer.y_pred)
            elif isinstance(self.explainer.y_pred, list):
                self.dataframe = self.dataframe.join(
                    pd.DataFrame(data=self.explainer.y_pred,
                                 columns=[self.predict_col],
                                 index=self.explainer.x_init.index)
                    )
            else:
                raise TypeError('y_pred must be of type pd.Series, pd.DataFrame or list')
        else:
            raise ValueError('y_pred must be set when calling compile function.')

        self.dataframe['_index_'] = self.explainer.x_init.index
        self.dataframe.rename(columns={f'{self.predict_col}': '_predict_'}, inplace=True)
        col_order = ['_index_', '_predict_'] + self.dataframe.columns.drop(['_index_', '_predict_']).tolist()
        random.seed(79)
        self.list_index = \
            random.sample(
                population=self.dataframe.index.tolist(),
                k=min(self.settings['rows'], len(self.dataframe.index.tolist()))
            )
        self.dataframe = self.dataframe[col_order].loc[self.list_index].sort_index()
        self.round_dataframe = self.dataframe.copy()
        for col in list(self.dataframe.columns):
            typ = self.dataframe[col].dtype
            if typ == float:
                std = self.dataframe[col].std()
                if std != 0:
                    digit = max(round(log10(1 / std) + 1) + 2, 0)
                    self.round_dataframe[col] = \
                        self.dataframe[col].map(f'{{:.{digit}f}}'.format).astype(float)

    def init_components(self):
        """
        Initialize components (graph, table, filter, settings, ...) and insert it inside
        components containers which are created by init_skeleton
        """

        self.components['settings']['input_rows'] = create_input_modal(
            id='rows',
            label="Number of rows for subset",
            tooltip="Set max number of lines for subset (datatable). \
                     Filter will be apply on this subset."
        )

        self.components['settings']['input_points'] = create_input_modal(
            id='points',
            label="Number of points for plot",
            tooltip="Set max number of points in feature contribution plots."
        )

        self.components['settings']['input_features'] = create_input_modal(
            id='features',
            label="Number of features to plot",
            tooltip="Set max number of features to plot in features \
                     importance and local explanation plots."
        )

        self.components['settings']['input_violin'] = create_input_modal(
            id='violin',
            label="Max number of labels for violin plot",
            tooltip="Set max number of labels to display a violin plot \
                     for feature contribution plot (otherwise a scatter \
                                                    plot is displayed)."
        )

        self.components['settings']['name'] = dbc.Row(
            [
                dbc.Checklist(
                    options=[{"label": "Use domain name for \
                              features name.", "value": 1}], value=[], inline=True,
                    id="name",
                    style={"margin-left": "20px"}
                ),
                dbc.Tooltip("Replace technical feature names by \
                            domain names if exists.",
                            target='name', placement='bottom'),
            ], className="g-3",
        )

        self.components['settings']['modal'] = dbc.Modal(
            [
                dbc.ModalHeader("Settings"),
                dbc.ModalBody(
                    dbc.Form(
                        [
                            self.components['settings']['input_rows'],
                            self.components['settings']['input_points'],
                            self.components['settings']['input_features'],
                            self.components['settings']['input_violin'],
                            self.components['settings']['name']
                        ]
                    )
                ),
                dbc.ModalFooter(
                    dbc.Button("Apply", id="apply", className="ml-auto")
                ),
            ],
            id="modal"
        )

        self.components['menu'] = dbc.Row(
            [
                dbc.Col(
                    [
                        html.Div(
                            daq.BooleanSwitch(
                                id='bool_groups',
                                on=True,
                                style={'display': 'none'} if self.explainer.features_groups is None else {},
                                color=self.color[0],
                                label={
                                    'label': 'Groups',
                                    'style': {
                                        'fontSize': 18,
                                        'color': self.color[0],
                                        'fontWeight': 'bold',
                                        "margin-left": "5px"
                                    },
                                },
                                labelPosition="right"
                            ),
                            style={"margin-right": "35px"}
                        )
                    ],
                    width="auto", align="center",
                ),
                dbc.Col(
                    [
                        html.H4(
                            [dbc.Badge("Regression", id='regression_badge',
                                       style={"margin-right": "5px",
                                              "margin-left": "0px"},
                                       color=''),
                             dbc.Badge("Classification", id='classification_badge', color='')
                             ], style={"margin-right": "5px"}
                        ),
                    ],
                    width="auto", align="center", style={'padding': 'auto'}
                ),
                dbc.Col(
                    dbc.Collapse(
                        dbc.Row(
                            [
                                # 2 columns to have class beside the dropdown buttons
                                dbc.Col([
                                    dbc.Label("Class:", style={'color': 'white', 'margin': '0px'}),
                                ], align="center"),
                                dbc.Col([
                                    dcc.Dropdown(
                                        id="select_label",
                                        options=[], value=None,
                                        clearable=False, searchable=False,
                                        style={"verticalAlign": "middle",
                                               "zIndex": '1010',
                                               "min-width": '160px',
                                               'height': '100%'}
                                    )
                                ], style={"margin-right": "17px",
                                          "padding": "0px",
                                          "width": "auto"})
                            ],
                            style={"margin": "0px"}
                        ),
                        is_open=True, id='select_collapse'
                    ),
                    width="auto", align="center", style={'padding': '0px'}
                ),
                dbc.Col([
                    html.Div(
                        [
                            html.Img(id='settings', title='settings', alt='Settings',
                                     src=self.app.get_asset_url('settings.png'),
                                     height='40px',
                                     style={'cursor': 'pointer'}),
                            self.components['settings']['modal'],
                        ]
                    )],
                    align="center", width="auto", style={'padding': '0px'}
                )
            ],
            className="g-0", justify="end"
        )

        self.adjust_menu()

        self.components['table']['dataset'] = dash_table.DataTable(
            id='dataset',
            data=self.round_dataframe.to_dict('records'),
            tooltip_data=[
                {
                    column: {'value': str(value), 'type': 'text'}
                    for column, value in row.items()
                } for row in self.dataframe.to_dict('rows')
            ], tooltip_duration=2000,

            columns=[{"name": '_index_', "id": '_index_'},
                     {"name": '_predict_', "id": '_predict_'}] +
                    [{"name": i, "id": i} for i in self.explainer.x_init],
            editable=False, row_deletable=False,
            style_as_list_view=True,
            virtualization=True,
            page_action='none',
            fixed_rows={'headers': True, 'data': 0},
            fixed_columns={'headers': True, 'data': 0},
            sort_action='custom', sort_mode='multi', sort_by=[],
            active_cell={'row': 0, 'column': 0, 'column_id': '_index_'},
            style_table={'overflowY': 'auto', 'overflowX': 'auto'},
            style_header={'height': '30px'},
            style_cell={
                'minWidth': '70px', 'width': '120px', 'maxWidth': '200px',
            },
        )

        self.components['graph']['global_feature_importance'] = MyGraph(
            figure=go.Figure(), id='global_feature_importance'
        )

        self.components['graph']['feature_selector'] = MyGraph(
            figure=go.Figure(), id='feature_selector'
        )

        # Component for the graph prediction picking
        self.components['graph']['prediction_picking'] = MyGraph(
            figure=go.Figure(), id='prediction_picking'
        )

        self.components['graph']['detail_feature'] = MyGraph(
            figure=go.Figure(), id='detail_feature'
        )

        # Component create to filter the dataset
        self.components['filter']['filter_dataset'] = dbc.Col(
            [dbc.Row(
                html.Div(
                    id='main',
                    children=[
                        html.Div(
                                id='filters',
                                children=[
                                    # Create Add Filter button
                                    dbc.Button(
                                        id='add_dropdown_button',
                                        children='Add Filter',
                                        color='warning',
                                        size='sm',
                                        style={'margin-right': '20px'}
                                    ),
                                    # Create reset Filter button (disabled of no filters applied)
                                    dbc.Button(
                                        id='reset_dropdown_button',
                                        children='Reset all existing filters',
                                        color='warning', disabled=True,
                                        size='sm',
                                        style={'margin-right': '20px'}
                                    ),
                                    # Create explanation button
                                    dbc.Button(
                                        "?",
                                        id="open_filter",
                                        size='sm',
                                        color="warning",
                                        ),
                                    # Create popover on the explanation button
                                    dbc.Popover(
                                        "Click here to know how you can apply filters.",
                                        target="open_filter",
                                        body=True,
                                        trigger="hover",
                                    ),
                                    # Modal associated to the explanation button
                                    dbc.Modal(
                                           [
                                            dbc.ModalHeader(
                                                dbc.ModalTitle("Filters explanation")
                                                ),
                                            dbc.ModalBody([
                                                html.Div(
                                                    dcc.Markdown(
                                                        self.explanations.filter
                                                        )
                                                    )
                                               ]),
                                            dbc.ModalFooter(
                                                dbc.Button(
                                                    "Close",
                                                    id="close_filter",
                                                    color="warning"
                                                    )
                                               ),
                                           ],
                                           id="modal_filter",
                                           centered=True,
                                           size='lg'
                                              ),
                                    # Div which will contains the filters
                                    html.Div(
                                        id='dropdowns_container',
                                        children=[]
                                    )
                                ]
                            )
                        ]
                    )
                ),
                dbc.Row(
                    html.Div([
                        html.Br(),
                        # Create Apply Filter button (Hidden if no filter to apply)
                        dbc.Button(
                                id='apply_filter',
                                children='Apply filters',
                                color='warning',
                                size='sm',
                                style={'display': 'none'}
                                )
                            ],
                        )
                    )
                ], style={'maxheight': '22rem', 'height': '21rem', 'zIndex': 800}
            )

        self.components['filter']['index'] = dbc.Col(dbc.Row(
            [
                dbc.Label("Index", align="center", width=4),
                dbc.Col([
                    dbc.Input(
                        id="index_id", type="text", size="s", placeholder="Id must exist",
                        debounce=True, persistence=True, style={'textAlign': 'right'}
                    )], width={"size": 5},
                    style={'padding': "0px"}
                ),
                dbc.Col([
                    html.Img(id='validation', alt='Validate', title='Validate index',
                             src=self.app.get_asset_url('reload.png'),
                             height='30px', style={'cursor': 'pointer'},
                             )], width={"size": 2},
                        style={'padding': "0px"}, align="center"
                        )
            ])
        )

        self.components['filter']['threshold'] = dbc.Col(
            [
                dbc.Label("Threshold", html_for="slider", id='threshold_label'),
                dcc.Slider(
                    min=0, max=self.max_threshold, value=0, step=0.1,
                    marks={f'{round(self.max_threshold * mark / 4)}': f'{round(self.max_threshold * mark / 4)}'
                           for mark in range(5)},
                    id="threshold_id",
                )
            ],
            className='filter_dashed'
        )

        self.components['filter']['max_contrib'] = dbc.Col(
            [
                dbc.Label(
                    "Features to display: ", id='max_contrib_label'),
                dcc.Slider(
                    min=1, max=min(self.settings['features'], len(self.dataframe.columns) - 2),
                    step=1, value=min(self.settings['features'], len(self.dataframe.columns) - 2),
                    id="max_contrib_id",
                )
            ],
            className='filter_dashed'
        )

        self.components['filter']['positive_contrib'] = dbc.Col(
            [dbc.Row(
                [
                    dbc.Label("Contributions to display:", style={'font-size': '95%'}),
                ]
            ),
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.Checklist(
                                options=[{"label": "Positive", "value": 1}], value=[1], inline=True,
                                id="check_id_positive",
                                # define the font-size style
                                style={'font-size': '82%'}
                            ), style={'display': 'inline-block'}
                        ),
                        dbc.Col(
                            dbc.Checklist(
                                options=[{"label": "Negative", "value": 1}], value=[1], inline=True,
                                id="check_id_negative",
                                # define the font-size style
                                style={'font-size': '82%'}
                                ), style={'display': 'inline-block'}, align="center"
                            )
                    ], className="g-0", justify="center"
                )
            ],
            className='filter_dashed'
        )

        self.components['filter']['masked_contrib'] = dbc.Col(
            [
                dbc.Label(
                    "Feature(s) to mask:"),
                dcc.Dropdown(options=[{'label': key, 'value': value} for key, value in sorted(
                    self.explainer.inv_features_dict.items(), key=lambda item: item[0])],
                    value='', multi=True, searchable=True,
                    id="masked_contrib_id"
                ),
            ],
            className='filter_dashed'
        )

    def make_skeleton(self):
        """
        Describe the app skeleton (bootstrap grid) and initialize components containers
        """
        self.skeleton['navbar'] = dbc.Container(
            [
                dbc.Row([
                        dbc.Col(
                            html.A(
                                dbc.Row([
                                    dbc.Col([
                                        html.Img(src=self.logo, height="40px")], className='col-1'),
                                    dbc.Col([
                                        html.H4("Shapash Monitor", id="shapash_title")]),
                                    ],
                                    align="center", style={'color': self.title_menu_color}
                                ),
                                href="https://github.com/MAIF/shapash", target="_blank",
                            ),
                            # Change md=3 to md=2
                            md=2, align="center", width="100%", style={'padding': 'auto'}
                        ),
                        dbc.Col([
                            html.A(
                                dbc.Row([
                                        html.H3(truncate_str(self.explainer.title_story, maxlen=40),
                                                id="shapash_title_story",
                                                style={'text-align': 'center'})]
                                        ),
                                href="https://github.com/MAIF/shapash", target="_blank",
                            )],
                            # Change md=3 to md=4
                            md=4, align="center", width="100%", style={'padding': 'auto'}
                        ),
                        dbc.Col([
                            self.components['menu']
                            ], align="end", md=6, width='100%'
                        )
                        ],
                        style={'padding': "5px 15px",
                               "verticalAlign": "middle",
                               "width": "auto",
                               "justify": "end"}
                        )
            ],
            fluid=True, style={'height': '100%', 'backgroundColor': self.bkg_color
                               }
        )

        self.skeleton['body'] = dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Card([
                                        html.Div(
                                         # To drow the global_feature_importance graph
                                         self.draw_component('graph', 'global_feature_importance'),
                                         id="card_global_feature_importance",
                                         # Position must be absolute to add the explanation button
                                         style={"position": 'absolute'}
                                         ),
                                        html.Div([
                                             # Create explanation button on feature importance graph
                                             dbc.Button(
                                                 "?",
                                                 id="open_feature_importance",
                                                 size='sm',
                                                 color="warning"
                                                 ),
                                             # Create popover for this button
                                             dbc.Popover(
                                                  "Click here to have more \
                                                  information on Feature Importance graph.",
                                                  target="open_feature_importance",
                                                  body=True,
                                                  trigger="hover",
                                                     ),
                                             # Create modal associated to this button
                                             dbc.Modal([
                                                        # Modal title
                                                        dbc.ModalHeader(
                                                            dbc.ModalTitle("Feature importance")
                                                            ),
                                                        dbc.ModalBody([
                                                            html.Div(
                                                                # Add explanation
                                                                dcc.Markdown(
                                                                    self.explanations.feature_importance
                                                                    )
                                                                    ),
                                                            # Here to add link in the modal
                                                            html.A('Click here for more details',
                                                                   href="https://github.com/MAIF/shapash/blob/master/tutorial/plot/tuto-plot03-features-importance.ipynb",
                                                                   # open new brother tab
                                                                   target="_blank",
                                                                   style={'color': self.color[0]})
                                                        ]),
                                                        # button to close the modal
                                                        dbc.ModalFooter(
                                                            dbc.Button(
                                                                "Close",
                                                                id="close_feature_importance",
                                                                color="warning"
                                                            )
                                                        ),
                                                    ],
                                                    id="modal_feature_importance",
                                                    centered=True,
                                                    size='lg'
                                                )
                                                ],
                                                # position must be relative
                                                style={'position': 'relative', 'left': '96%'})
                                    ])
                            ],
                            md=5,
                            style={'padding': '10px 10px 0px 10px'},
                        ),
                        dbc.Col(
                            [
                                # Tabs that contain 3 children tab (Dataset,
                                # Dataset Filters and True Value Vs Pedicted Values)
                                dbc.Tabs([
                                    # Tab which contains the datatable component
                                    dbc.Tab(
                                        # draw datatable component
                                        self.draw_component('table', 'dataset'),
                                        # Tab name
                                        label='Dataset',
                                        className="card",
                                        id='card_dataset',
                                        # Style of the tab
                                        style={'cursor': 'pointer'},
                                        label_style={'color': "black", 'height': '30px',
                                                     'padding': '0px 5px'},
                                        # Style when the tab is activated
                                        active_tab_class_name="fw-bold fst-italic",
                                        active_label_style={'border-top': '3px solid',
                                                            'border-top-color': self.color[0]
                                                            }
                                        ),
                                    # Tab which contains components to filter the dataset
                                    dbc.Tab(
                                        dbc.Card(
                                            dbc.CardBody(
                                                html.Div(
                                                    # draw the component
                                                    self.draw_filter_table(),
                                                    id='card_filter_dataset',
                                                    # To add scroll in overflow y
                                                    style={'overflow-y': "scroll",
                                                           'overflow-x': 'hidden'})
                                            ), style={'height': '24.1rem'},
                                        ),
                                        # Tab name
                                        label='Dataset Filters',
                                        # Style of the tab
                                        label_style={'color': "black", 'height': '30px',
                                                     'padding': '0px 5px'},
                                        tab_style={'border-left': '2px solid #ddd',
                                                   'border-right': '2px solid #ddd'},
                                        # Style when the tab is activated
                                        active_tab_class_name="fw-bold fst-italic",
                                        active_label_style={'border-top': '3px solid',
                                                            'border-top-color': self.color[0]
                                                            }
                                         ),
                                    # Tab which contains prediction picking graph
                                    # and its explanation button
                                    dbc.Tab(
                                        dbc.Card([
                                            html.Div(
                                                # draw prediction picking graph
                                                self.draw_component('graph', 'prediction_picking'),
                                                id="card_prediction_picking",
                                                # Position must be absolute to add
                                                # the explanation button
                                                style={"position": 'absolute'}
                                            ),
                                            html.Div([
                                                 # Create explanation button
                                                 dbc.Button(
                                                     "?",
                                                     id="open_prediction_picking",
                                                     size='sm',
                                                     color="warning"
                                                     ),
                                                 # Create popover for this button
                                                 dbc.Popover(
                                                         "Click here to have more \
                                                         information on Prediction Picking graph.",
                                                         target="open_prediction_picking",
                                                         body=True,
                                                         trigger="hover",
                                                     ),
                                                 # Create modal associated to this button
                                                 dbc.Modal([
                                                             # Modal title
                                                             dbc.ModalHeader(
                                                                dbc.ModalTitle("True values Vs Predicted values")
                                                              ),
                                                             dbc.ModalBody([
                                                                    html.Div(
                                                                        # explanation
                                                                        dcc.Markdown(self.explanations.prediction_picking)
                                                                        ),
                                                                    # Here to add link in the modal
                                                                    html.Div(html.Img(src="https://github.com/MAIF/shapash/blob/master/docs/_static/shapash_select_subset.gif?raw=true")),
                                                                    # Here to add a link in the modal
                                                                    html.A('Click here for more details',
                                                                           href="https://github.com/MAIF/shapash/blob/master/tutorial/plot/tuto-plot06-prediction_plot.ipynb",
                                                                           # open new brother tab
                                                                           target="_blank",
                                                                           style={'color': self.color[0]})
                                                                    ]),
                                                             dbc.ModalFooter(
                                                                 # button to close the modal
                                                                 dbc.Button(
                                                                     "Close",
                                                                     id="close_prediction_picking",
                                                                     color="warning"
                                                                 )
                                                              ),
                                                        ],
                                                        id="modal_prediction_picking",
                                                        centered=True,
                                                        size='lg'
                                                    )
                                                 # Position must be relative
                                                ],  style={'position': 'relative', 'left': '97%'})
                                         ]),
                                        # Tab name
                                        label='True Values Vs Predicted Values',
                                        # Style of the tab
                                        label_style={'color': "black", 'height': '30px',
                                                     'padding': '0px 5px'},
                                        # Style when the tab is activated
                                        active_tab_class_name="fw-bold fst-italic",
                                        active_label_style={'border-top': '3px solid',
                                                            'border-top-color': self.color[0]
                                                            }
                                    )
                                ], id="tabs"
                                )
                            ],
                            md=7,
                            style={'padding': '10px 10px'},
                        ),
                    ], style={'padding': '10px 10px 0px 10px',
                              'height': '100%'}, align="center"
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                # Card which contains feature selector graph
                                # and explanation button
                                dbc.Card([
                                        html.Div(
                                            # Draw feature selector graph
                                            self.draw_component('graph', 'feature_selector'),
                                            id='card_feature_selector',
                                            # Position must be absolute to
                                            # add explanation button
                                            style={"position": 'absolute'}
                                             ),
                                         html.Div([
                                             # Create explanation button
                                             dbc.Button(
                                                 "?",
                                                 id="open_feature_selector",
                                                 size='sm',
                                                 color="warning"
                                                 ),
                                             # popover of this button
                                             dbc.Popover(
                                                         "Click here to have more \
                                                         information on Feature Selector graph.",
                                                         target="open_feature_selector",
                                                         body=True,
                                                         trigger="hover",
                                                     ),
                                             # Modal of this button
                                             dbc.Modal([
                                                        dbc.ModalHeader(
                                                            dbc.ModalTitle("Feature selector")
                                                            ),
                                                        dbc.ModalBody([
                                                            html.Div(
                                                                # explanations
                                                                dcc.Markdown(self.explanations.feature_selector)
                                                                ),
                                                            # Here to add link
                                                            html.A('Click here for more details',
                                                                   href="https://github.com/MAIF/shapash/blob/master/tutorial/plot/tuto-plot02-contribution_plot.ipynb",
                                                                   # open new brother tab
                                                                   target="_blank",
                                                                   style={'color': self.color[0]})
                                                                ]),
                                                        dbc.ModalFooter(
                                                            # Button to close modal
                                                            dbc.Button(
                                                                "Close",
                                                                id="close_feature_selector",
                                                                color="warning"
                                                            )
                                                        ),
                                                    ],
                                                    id="modal_feature_selector",
                                                    centered=True,
                                                    size='lg'
                                                )
                                                ],
                                                 # position must be relative
                                                 style={'position': 'relative',
                                                        'left': '96%'})
                                          ])
                            ],
                            md=5,
                            align="center",
                            style={'padding': '0px 10px'},
                        ),
                        dbc.Col(
                            [
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                             # Card that contains detail feature graph
                                             # and explanation button
                                             dbc.Card([
                                                 html.Div(
                                                     # draw detail_feature graph
                                                     self.draw_component('graph', 'detail_feature'),
                                                     id='card_detail_feature',
                                                     style={"position": 'absolute'}
                                                 ),
                                                 html.Div([
                                                     # Create explanation button
                                                     dbc.Button(
                                                         "?",
                                                         id="open_detail_feature",
                                                         size='sm',
                                                         color="warning"
                                                         ),
                                                     # Popover of this button
                                                     dbc.Popover(
                                                         "Click here to have more \
                                                         information on Detail Feature graph.",
                                                         target="open_detail_feature",
                                                         body=True,
                                                         trigger="hover",
                                                     ),
                                                     # Modal of this button
                                                     dbc.Modal(
                                                            [
                                                             dbc.ModalHeader(
                                                                 dbc.ModalTitle("Detail feature")
                                                                 ),
                                                             dbc.ModalBody([
                                                                 html.Div(
                                                                     # explanations
                                                                     dcc.Markdown(self.explanations.detail_feature)
                                                                     ),
                                                                 # Here to add link on the modal
                                                                 html.A('Click here for more details',
                                                                        href="https://github.com/MAIF/shapash/blob/master/tutorial/plot/tuto-plot01-local_plot-and-to_pandas.ipynb",
                                                                        # open new brother tab
                                                                        target="_blank",
                                                                        style={'color': self.color[0]})
                                                                ]),
                                                             dbc.ModalFooter(
                                                                 # Button to close the modal
                                                                 dbc.Button(
                                                                     "Close",
                                                                     id="close_detail_feature",
                                                                     color="warning"
                                                                     )
                                                                ),
                                                            ],
                                                            id="modal_detail_feature",
                                                            centered=True,
                                                            size='lg'
                                                        )

                                                        ],
                                                          # Position must be relative
                                                          style={
                                                            'position': 'relative',
                                                            'left': '96%'
                                                            })
                                                ])
                                            ],
                                            md=8,
                                            align="center",
                                        ),
                                        dbc.Col(
                                            [
                                                html.Div(
                                                    self.draw_filter(),
                                                    className="card_filter",
                                                    id='card_filter',
                                                ),
                                            ],
                                            md=4,
                                            align="center",
                                        ),
                                    ],
                                ),
                            ],
                            md=7,
                            align="center",
                            style={'padding': '0px 10px'},
                        ),
                    ],
                    style={'padding': '10px 5px 10px 10px'},
                ),
            ],
            className="mt-12",
            fluid=True,
            # To drop the x scroll-bar
            style={'overflow-x': 'hidden'}
        )

    def adjust_menu(self):
        """
        Override menu from explainer object depending on
        classification or regression case.
        """
        on_style = {'backgroundColor': self.color[0],
                    'color': self.bkg_color,
                    'margin-right': '0.5rem'}
        off_style = {'backgroundColor': self.color[1],
                     'color': self.bkg_color,
                     'margin-right': '0.5rem'}
        if self.explainer._case == 'classification':
            self.components['menu']['select_label'].options = \
                [
                    {'label': f'{self.explainer.label_dict[label] if self.explainer.label_dict else label}',
                     'value': label}
                    for label in self.explainer._classes
            ]
            self.components['menu']['classification_badge'].style = on_style
            self.components['menu']['regression_badge'].style = off_style
            self.components['menu']['select_label'].value = self.label

        elif self.explainer._case == 'regression':
            self.components['menu']['classification_badge'].style = off_style
            self.components['menu']['regression_badge'].style = on_style
            self.components['menu']['select_collapse'].is_open = False

        else:
            raise ValueError(f'No rule defined for explainer case : {self.explainer._case}')

    def draw_component(self,
                       component_type,
                       component_id,
                       title=None):
        """
        Method which return a component from a type and id.
        It's the method to insert component inside component container.
        Parameters
        ----------
        component_type : string
            Type of the component. Can be table, graph, ...
        component_id : string
            Id of the component. It must be unique.
        title : string, optional
            by default None
        Returns
        -------
        list
            list of components
            (combining for example Graph + embed button to get fullscreen
             details)
        """
        component = [html.H4(title)] if title else []
        component.append(self.components[component_type][component_id])
        component.append(
            html.A(
                html.I("fullscreen",
                       className="material-icons tiny",
                       style={'marginTop': '8px', 'marginLeft': '1px'}
                       ),
                id=f"ember_{component_id}",
                className="dock-expand",
                **{'data-component-type': component_type},
                # Get components'id
                **{'data-component-id': component_id}
            )
        )
        return component
    
    def draw_filter_table(self):
        """
        Method which returns the filter dataset components block.
        Returns
        -------
            component
        """
        return self.components['filter']['filter_dataset']
    
    def draw_filter(self):
        """
        Method which returns filter components block for local
        contributions plot.
        Returns
        -------
        list
            list of components
        """
        filter = [
            dbc.Container(
                [
                    dbc.Row([self.components['filter']['index']],
                            align="center", style={"height": "4rem"}
                            ),
                    dbc.Row([self.components['filter']['threshold']],
                            align="center", style={"height": "5rem"}
                            ),
                    dbc.Row([self.components['filter']['max_contrib']],
                            align="center", style={"height": "5rem"}
                            ),
                    dbc.Row([self.components['filter']['positive_contrib']],
                            align="center", style={"height": "4rem"}
                            ),
                    dbc.Row([self.components['filter']['masked_contrib']],
                            align="center"),
                ],
            ),
        ]
        return filter

    def select_point(self,
                     graph,
                     click_data):
        """
        Method which set the selected point in graph component
        corresponding to click_data.
        """
        if click_data:
            curve_id = click_data['points'][0]['curveNumber']
            point_id = click_data['points'][0]['pointIndex']
            for curve in range(
                    len(self.components['graph'][graph].figure['data'])):
                self.components['graph'][graph].figure['data'][curve].selectedpoints = \
                    [point_id] if curve == curve_id else []

    def callback_fullscreen_buttons(self):
        """
        Initialize callbacks for each fullscreen button
        the callback alter style of the component (height, ...)
        Returns
        -------
        dict
            Style of the component
        """
        app = self.app
        components_to_init = dict([(graph, 'graph') for graph in self.components['graph'].keys()])
        components_to_init['dataset'] = 'table'
        for component_id, component_type in components_to_init.items():
            component_property = 'style' if component_type == "graph" else "style_table"

            @app.callback(
                [
                    Output(f'card_{component_id}', 'style'),
                    Output(f'{component_id}', component_property),
                ],
                [
                    Input(f'ember_{component_id}', 'n_clicks'),
                    Input(f'ember_{component_id}', 'data-component-type'),
                    Input(f'ember_{component_id}', 'data-component-id')
                ]
            )
            def ember(click,
                      data_component_type,
                      data_component_id):
                """
                Function used to set style of cards and components.
                Prediction picking graph style is different than the other
                graph because it is placed in a tab.
                ---------------------------------------------------------------
                click: click on zoom button
                data_component_type: component type
                data_component_id: component id
                --------------------------------------------------------------
                return style of cards and style of components
                """
                click = 2 if click is None else click
                toggle_on = True if click % 2 == 0 else False
                if toggle_on:
                    # Style for graph
                    style_component = {
                        'height': '21.6rem'
                    }
                    this_style_card = {
                        'height': '22rem', 'zIndex': 900,
                    }
                    # Style for prediction picking graph
                    if data_component_id == 'prediction_picking':
                        style_component = {
                            'height': '20.6rem',
                        }
                        this_style_card = {
                            'height': '20.8rem', 'zIndex': 901,
                        }
                    # Style for the Dataset
                    if data_component_type == 'table':
                        style_component = {
                            'maxHeight': '23rem',
                        }
                        this_style_card = {
                            'height': '24.1rem', 'zIndex': 900,
                        }
                    return this_style_card, style_component

                else:
                    # Style when zoom button is clicked
                    this_style_card = {
                        'height': '70vh',
                        'width': 'auto',
                        'zIndex': 998,
                        'position': 'fixed', 'top': '55px',
                        'bottom': 0, 'left': 0, 'right': 0,
                    }
                    style_component = {
                        'height': '89vh', 'maxHeight': '89vh',
                    }
                    return this_style_card, style_component

    def init_callback_settings(self):
        app = self.app
        self.components['settings']['input_rows']['rows'].value = self.settings['rows']
        self.components['settings']['input_points']['points'].value = self.settings['points']
        self.components['settings']['input_features']['features'].value = self.settings['features']
        self.components['settings']['input_violin']['violin'].value = self.settings['violin']

        for id in self.settings.keys():
            @app.callback(
                [Output(f'{id}', 'valid'),
                 Output(f'{id}', 'invalid')],
                [Input(f'{id}', "value")]
            )
            def update_valid(value):
                """
                actualise valid and invalid icon in input component
                Parameters
                ----------
                value : int
                    value of input component
                Returns
                -------
                    tuple of boolean
                """
                patt = re.compile('^[0-9]*[1-9][0-9]*$')
                if patt.match(str(value)):
                    return True, False
                else:
                    return False, True

        @app.callback(
            Output("modal", "is_open"),
            [
                Input("settings", "n_clicks"),
                Input("apply", "n_clicks")],
            [
                State('rows', 'valid'),
                State('points', 'valid'),
                State('features', 'valid'),
                State('violin', 'valid'),
            ],
        )
        def toggle_modal(n1,
                         n2,
                         rows,
                         points,
                         features,
                         violin):
            """
            open modal /close modal (only if all input are valid)
            """
            ctx = dash.callback_context
            if ctx.triggered[0]['prop_id'] == 'settings.n_clicks':
                if n1 is not None:
                    return True
            else:
                if n2 is not None:
                    if all([rows, points, features, violin]):
                        return False
                    else:
                        return True
            return False

    def callback_generator(self):
        app = self.app

        @app.callback(
            [
                Output('dataset', 'data'),
                Output('dataset', 'tooltip_data'),
                Output('dataset', 'columns'),
                Output('dataset', 'active_cell'),
            ],
            [
                Input('prediction_picking', 'selectedData'),
                Input('modal', 'is_open'),
                Input('apply_filter', 'n_clicks'),
                Input('reset_dropdown_button', 'n_clicks'),
                Input({'type': 'del_dropdown_button', 'index': ALL}, 'n_clicks')
            ],
            [
                State('rows', 'value'),
                State('name', 'value'),
                State({'type': 'var_dropdown', 'index': ALL}, 'value'),
                State({'type': 'var_dropdown', 'index': ALL}, 'id'),
                State({'type': 'dynamic-str', 'index': ALL}, 'value'),
                State({'type': 'dynamic-str', 'index': ALL}, 'id'),
                State({'type': 'dynamic-bool', 'index': ALL}, 'value'),
                State({'type': 'dynamic-bool', 'index': ALL}, 'id'),
                State({'type': 'dynamic-date', 'index': ALL}, 'start_date'),
                State({'type': 'dynamic-date', 'index': ALL}, 'end_date'),
                State({'type': 'dynamic-date', 'index': ALL}, 'id'),
                State({'type': 'lower', 'index': ALL}, 'value'),
                State({'type': 'lower', 'index': ALL}, 'id'),
                State({'type': 'upper', 'index': ALL}, 'value'),
                State({'type': 'upper', 'index': ALL}, 'id'),
                State('dropdowns_container', 'children')
            ]
        )
        def update_datatable(selected_data,
                             is_open,
                             nclicks_apply,
                             nclicks_reset,
                             nclicks_del,
                             rows,
                             name,
                             val_feature,
                             id_feature,
                             val_str_modality,
                             id_str_modality,
                             val_bool_modality,
                             id_bool_modality,
                             start_date,
                             end_date,
                             id_date,
                             val_lower_modality,
                             id_lower_modality,
                             val_upper_modality,
                             id_upper_modality,
                             children):
            """
            This function is used to update the datatable according to sorting,
            filtering and settings modifications.
            ------------------------------------------------------------------
            selected_data: selected data in prediction picking graph
            is_open: modal
            nclicks_apply: click on Apply Filter button
            nclicks_reset: click on Reset All Filter button
            nclicks_del: click on delete button
            rows: number of rows for subset
            name: name for features name
            val_feature: feature selected to filter
            id_feature: id of feature selected to filter
            val_str_modality: string modalities selected
            id_str_modality: id of string modalities selected
            val_bool_modality: boolean modalities selected
            id_bool_modality: id of boolean modalities selected
            start_date: start dates selected
            end_date: end dates selected
            id_date: id of dates selected
            val_lower_modality: lower values of numeric filter
            id_lower_modality: id of lower modalities of numeric filter
            val_upper_modality: upper values of numeric filter
            id_upper_modality: id of upper values of numeric filter
            children: children of dropdown container
            ------------------------------------------------------------------
            return
            data: available dataset
            tooltip_data: tooltip of the dataset
            columns: columns of the dataset
            active_cell: activated cell
            """
            ctx = dash.callback_context
            active_cell = no_update
            df = self.round_dataframe
            columns = self.components['table']['dataset'].columns
            if ctx.triggered[0]['prop_id'] == 'modal.is_open':
                if is_open:
                    raise PreventUpdate
                else:
                    self.settings['rows'] = rows
                    self.init_data()
                    active_cell = {'row': 0, 'column': 0, 'column_id': '_index_'}
                    self.settings_ini['rows'] = self.settings['rows']
                    if name == [1]:
                        columns = [
                            {"name": '_index_', "id": '_index_'},
                            {"name": '_predict_', "id": '_predict_'}] + \
                            [{"name": self.explainer.features_dict[i], "id": i} for i in self.explainer.x_init]
                    df = self.round_dataframe
            elif ((ctx.triggered[0]['prop_id'] == 'prediction_picking.selectedData') and
                  (selected_data is not None) and (len(selected_data) > 1)):
                row_ids = []
                # If some data have been selected in prediction picking graph
                if selected_data is not None and len(selected_data) > 1:
                    for p in selected_data['points']:
                        row_ids.append(p['customdata'])
                    df = self.round_dataframe.loc[row_ids]
                else:
                    df = self.round_dataframe
            # If click on reset button
            elif ctx.triggered[0]['prop_id'] == 'reset_dropdown_button.n_clicks':
                df = self.round_dataframe
            # If click on Apply filter
            elif ((ctx.triggered[0]['prop_id'] == 'apply_filter.n_clicks') | (
                    (ctx.triggered[0]['prop_id'] == 'prediction_picking.selectedData') and
                  ((selected_data is None))) | (
                    (ctx.triggered[0]['prop_id'] == 'prediction_picking.selectedData') and
                  (selected_data is not None and len(selected_data) == 1 and selected_data['points'][0]['curveNumber'] > 0)
                  )):
                # get list of ID
                feature_id = [id_feature[i]['index'] for i in range(len(id_feature))]
                str_id = [id_str_modality[i]['index'] for i in range(len(id_str_modality))]
                bool_id = [id_bool_modality[i]['index'] for i in range(len(id_bool_modality))]
                lower_id = [id_lower_modality[i]['index'] for i in range(len(id_lower_modality))]
                date_id = [id_date[i]['index'] for i in range(len(id_date))]
                df = self.round_dataframe
                # If there is some filters
                if len(feature_id) > 0:
                    for i in range(len(feature_id)):
                        # String filter
                        if feature_id[i] in str_id:
                            position = np.where(np.array(str_id) == feature_id[i])[0][0]
                            if ((position is not None) & (val_str_modality[position] is not None)):
                                df = df[df[val_feature[i]].isin(val_str_modality[position])]
                            else:
                                df = df
                        # Boolean filter
                        elif feature_id[i] in bool_id:
                            position = np.where(np.array(bool_id) == feature_id[i])[0][0]
                            if ((position is not None) & (val_bool_modality[position] is not None)):
                                df = df[df[val_feature[i]] == val_bool_modality[position]]
                            else:
                                df = df
                        # Date filter
                        elif feature_id[i] in date_id:
                            position = np.where(np.array(date_id) == feature_id[i])[0][0]
                            if((position is not None) &
                               (start_date[position] < end_date[position])):
                                df = df[((df[val_feature[i]] >= start_date[position]) &
                                         (df[val_feature[i]] <= end_date[position]))]
                            else:
                                df = df
                        # Numeric filter
                        elif feature_id[i] in lower_id:
                            position = np.where(np.array(lower_id) == feature_id[i])[0][0]
                            if((position is not None) & (val_lower_modality[position] is not None) &
                               (val_upper_modality[position] is not None)):
                                if (val_lower_modality[position] < val_upper_modality[position]):
                                    df = df[(df[val_feature[i]] >= val_lower_modality[position]) &
                                            (df[val_feature[i]] <= val_upper_modality[position])]
                                else:
                                    df = df
                            else:
                                df = df
                        else:
                            df = df
                else:
                    df = df
                if len(df) == 0:
                    raise ValueError(
                        "Your dataframe is empty. It must have at list one row"
                         )
            elif None not in nclicks_del:
                df = self.round_dataframe
            else:
                raise dash.exceptions.PreventUpdate
            self.components['table']['dataset'].data = df.to_dict('records')
            self.components['table']['dataset'].tooltip_data = [
                {
                    column: {'value': str(value), 'type': 'text'}
                    for column, value in row.items()
                } for row in df.to_dict('rows')
            ]
            return (
                self.components['table']['dataset'].data,
                self.components['table']['dataset'].tooltip_data,
                columns,
                active_cell,
            )

        @app.callback(
            [
                Output('global_feature_importance', 'figure'),
                Output('global_feature_importance', 'clickData')
            ],
            [
                Input('select_label', 'value'),
                Input('dataset', 'data'),
                Input('prediction_picking', 'selectedData'),
                Input('apply_filter', 'n_clicks'),
                Input('reset_dropdown_button', 'n_clicks'),
                Input({'type': 'del_dropdown_button', 'index': ALL}, 'n_clicks'),
                Input('modal', 'is_open'),
                Input('card_global_feature_importance', 'n_clicks'),
                Input('bool_groups', 'on'),
                Input('ember_global_feature_importance', 'n_clicks')
            ],
            [
                State('global_feature_importance', 'clickData'),
                State('features', 'value')
            ]
        )
        def update_feature_importance(label,
                                      data,
                                      selected_data,
                                      apply_filters,
                                      reset_filter,
                                      nclicks_del,
                                      is_open,
                                      n_clicks,
                                      bool_group,
                                      click_zoom,
                                      clickData,
                                      features):
            """
            update feature importance plot according label, click on graph,
            filters applied and subset selected in prediction picking graph.
            ------------------------------------------------------------
            label: label of data
            data: dataset
            selected_data : data selected on prediction picking graph
            apply_filters: click on apply filter button
            reset_filter: click on reset filter button
            nclicks_del: click on del button
            is_open: modal
            n_clicks: click on features importance card
            bool_group: display groups
            click_zoom: click on zoom button
            clickData: click on features importance graph
            features: features value
            -------------------------------------------------------------
            return
            figure of Features Importance graph
            click on Features Importance graph
            """
            ctx = dash.callback_context
            # Zoom is False by Default. It becomes True if we click on it
            click = 2 if click_zoom is None else click_zoom
            if click % 2 == 0:
                zoom_active = False
            else:
                zoom_active = True
            selection = None
            selected_feature = self.explainer.inv_features_dict.get(
                clickData['points'][0]['label'].replace('<b>', '').replace('</b>', '')
            ) if clickData else None
            if ctx.triggered[0]['prop_id'] == 'modal.is_open':
                if is_open:
                    raise PreventUpdate
                else:
                    self.settings['features'] = features
                    self.settings_ini['features'] = self.settings['features']
            elif ctx.triggered[0]['prop_id'] == 'select_label.value':
                self.label = label
                selection = None
            elif ctx.triggered[0]['prop_id'] == 'dataset.data':
                self.list_index = [d['_index_'] for d in data]
            elif ctx.triggered[0]['prop_id'] == 'bool_groups.on':
                clickData = None  # We reset the graph and clicks if we toggle the button
            # If we have selected data on prediction picking graph
            elif ((ctx.triggered[0]['prop_id'] == 'prediction_picking.selectedData') and
                  (selected_data is not None) and (len(selected_data) > 1)):
                row_ids = []
                if selected_data is not None and len(selected_data) > 1:
                    for p in selected_data['points']:
                        row_ids.append(p['customdata'])
                    selection = row_ids
                else:
                    selection = None
                #when group
                if self.explainer.features_groups and bool_group:
                    list_sub_features = [f for group_features in self.explainer.features_groups.values()
                                      for f in group_features]
                    if selected_feature not in list_sub_features:
                        selected_feature = None
            # If click on a single point on prediction picking, do nothing
            elif ((ctx.triggered[0]['prop_id'] == 'prediction_picking.selectedData') and
                  (selected_data is not None) and (len(selected_data) == 1)):
                # If there is some filters applied
                if (len([d['_index_'] for d in data]) != len(self.list_index)):
                    selection = [d['_index_'] for d in data]
                else:
                    selection = None
            # If we have dubble click on prediction picking to remove the selected subset
            elif ((ctx.triggered[0]['prop_id'] == 'prediction_picking.selectedData') and
                  (selected_data is None)):
                # If there is some filters applied
                if (len([d['_index_'] for d in data]) != len(self.list_index)):
                    selection = [d['_index_'] for d in data]
                else:
                    selection = None
                #when group
                if self.explainer.features_groups and bool_group:
                    list_sub_features = [f for group_features in self.explainer.features_groups.values()
                                      for f in group_features]
                    if selected_feature not in list_sub_features:
                        selected_feature = None
            # If we click on reset filter button
            elif ctx.triggered[0]['prop_id'] == 'reset_dropdown_button.n_clicks':
                selection = None
                #when group
                if self.explainer.features_groups and bool_group:
                    list_sub_features = [f for group_features in self.explainer.features_groups.values()
                                      for f in group_features]
                    if selected_feature not in list_sub_features:
                        selected_feature = None
            # If we click on Apply button
            elif ctx.triggered[0]['prop_id'] == 'apply_filter.n_clicks':
                selection = [d['_index_'] for d in data]
                #when group
                if self.explainer.features_groups and bool_group:
                    list_sub_features = [f for group_features in self.explainer.features_groups.values()
                                      for f in group_features]
                    if selected_feature not in list_sub_features:
                        selected_feature = None
            # If we click on the last del button
            elif (('del_dropdown_button' in ctx.triggered[0]['prop_id']) &
                  (None not in nclicks_del)):
                selection = None
                #when group
                if self.explainer.features_groups and bool_group:
                    list_sub_features = [f for group_features in self.explainer.features_groups.values()
                                      for f in group_features]
                    if selected_feature not in list_sub_features:
                        selected_feature = None
            elif (ctx.triggered[0]['prop_id'] == 'card_global_feature_importance.n_clicks'
                  and self.explainer.features_groups and bool_group):
                row_ids = []
                if selected_data is not None and len(selected_data) > 1:
                    # we plot prediction picking subset
                    for p in selected_data['points']:
                        row_ids.append(p['customdata'])
                    selection = row_ids
                elif (len([d['_index_'] for d in data]) != len(self.list_index)):
                    selection = [d['_index_'] for d in data]
                else:
                    selection = None
                # When we click twice on the same bar this will reset the graph
                if self.last_click_data == clickData:
                    selected_feature = None
                list_sub_features = [f for group_features in self.explainer.features_groups.values()
                                      for f in group_features]
                if selected_feature in list_sub_features:
                    self.last_click_data = clickData
                    raise PreventUpdate
                else:
                    pass
            else:
                # Zoom management to generate graph which have global axis
                if len(self.components['graph']['global_feature_importance'].figure['data']) == 1:
                    selection = None
                else:
                    row_ids = []
                    if selected_data is not None and len(selected_data) > 1:
                        # we plot prediction picking subset
                        for p in selected_data['points']:
                            row_ids.append(p['customdata'])
                        selection = row_ids
                    else:
                        # we plot filter subset
                        selection = [d['_index_'] for d in data]
                self.last_click_data = clickData

            group_name = selected_feature if (self.explainer.features_groups is not None
                                              and selected_feature in self.explainer.features_groups.keys()) else None

            self.components['graph']['global_feature_importance'].figure = \
                self.explainer.plot.features_importance(
                    max_features=features,
                    selection=selection,
                    label=self.label,
                    group_name=group_name,
                    display_groups=bool_group,
                    zoom=zoom_active
                )
            # Adjust graph with adding x axis title
            self.components['graph']['global_feature_importance'].adjust_graph(x_ax='Contribution')
            self.components['graph']['global_feature_importance'].figure.layout.clickmode = 'event+select'
            if selected_feature:
                if self.explainer.features_groups is None:
                    self.select_point('global_feature_importance', clickData)
                elif selected_feature not in self.explainer.features_groups.keys():
                    self.select_point('global_feature_importance', clickData)

            # font size can be adapted to screen size
            nb_car = max([len(self.components['graph']['global_feature_importance'].figure.data[0].y[i]) for i in
                          range(len(self.components['graph']['global_feature_importance'].figure.data[0].y))])
            self.components['graph']['global_feature_importance'].figure.update_layout(
                yaxis=dict(tickfont={'size': min(round(500 / nb_car), 12)})
            )

            self.last_click_data = clickData
            return self.components['graph']['global_feature_importance'].figure, clickData

        @app.callback(
            Output(component_id='feature_selector', component_property='figure'),
            [
                Input('global_feature_importance', 'clickData'),
                Input('prediction_picking', 'selectedData'),
                Input('dataset', 'data'),
                Input('apply_filter', 'n_clicks'),
                Input('reset_dropdown_button', 'n_clicks'),
                Input({'type': 'del_dropdown_button', 'index': ALL}, 'n_clicks'),
                Input('select_label', 'value'),
                Input('modal', 'is_open'),
                Input('ember_feature_selector', 'n_clicks')
            ],
            [
                State('points', 'value'),
                State('violin', 'value')
            ]
        )
        def update_feature_selector(feature,
                                    selected_data,
                                    data,
                                    apply_filters,
                                    reset_filter,
                                    nclicks_del,
                                    label,
                                    is_open,
                                    click_zoom,
                                    points,
                                    violin):
            """
            Update feature plot according to label, data,
            selected feature on features importance graph,
            filters and settings modifications
            --------------------------------------------
            feature: click on feature importance graph
            selected_data: Data selected on prediction picking graph
            data: dataset
            apply_filters: click on apply filter button
            reset_filter: click on reset filter button
            nclicks_del: click del button
            label: selected label
            is_open: modal
            click_zoom: click on zoom button
            points: points value in setting
            violin: violin value in setting
            ---------------------------------------------
            return
            figure: feature selector graph
            """
            # Zoom is False by Default. It becomes True if we click on it
            click = 2 if click_zoom is None else click_zoom
            if click % 2 == 0:
                zoom_active = False
            else:
                zoom_active = True  # To check if zoom is activated
            ctx = dash.callback_context
            if ctx.triggered[0]['prop_id'] == 'modal.is_open':
                if is_open:
                    raise PreventUpdate
                else:
                    self.settings['points'] = points
                    self.settings_ini['points'] = self.settings['points']
                    self.settings['violin'] = violin
                    self.settings_ini['violin'] = self.settings['violin']

            elif ctx.triggered[0]['prop_id'] == 'select_label.value':
                self.label = label
            elif ctx.triggered[0]['prop_id'] == 'global_feature_importance.clickData':
                if feature is not None:
                    # Removing bold
                    self.selected_feature = feature['points'][0]['label'].replace('<b>', '').replace('</b>', '')
                    if feature['points'][0]['curveNumber'] == 0 and \
                              len(self.components['graph']['global_feature_importance'].figure['data']) == 2:
                        if selected_data is not None and len(selected_data) > 1:
                            row_ids = []
                            for p in selected_data['points']:
                                row_ids.append(p['customdata'])
                            self.subset = row_ids
                        else:
                            self.subset = [d['_index_'] for d in data]
                    else:
                        self.subset = self.list_index
            # If we have selected data on prediction picking graph
            elif ((ctx.triggered[0]['prop_id'] == 'prediction_picking.selectedData') and
                  (selected_data is not None)):
                row_ids = []
                if selected_data is not None and len(selected_data) > 1:
                    for p in selected_data['points']:
                        row_ids.append(p['customdata'])
                    self.subset = row_ids
            # if we have click on reset button
            elif ctx.triggered[0]['prop_id'] == 'reset_dropdown_button.n_clicks':
                self.subset = None
            # If we have clik on Apply filter button
            elif ctx.triggered[0]['prop_id'] == 'apply_filter.n_clicks':
                self.subset = [d['_index_'] for d in data]
            # If we have click on the last del button
            elif (('del_dropdown_button' in ctx.triggered[0]['prop_id']) &
                  (None not in nclicks_del)):
                self.subset = None
            else:
                # Zoom management to generate graph which have global axis
                if len(self.components['graph']['global_feature_importance'].figure['data']) == 1:
                    self.subset = self.list_index
                elif (len(self.components['graph']['global_feature_importance'].figure['data']) == 2):
                    if feature is not None:
                        if feature['points'][0]['curveNumber'] == 0:
                            if selected_data is not None and len(selected_data) > 1:
                                row_ids = []
                                for p in selected_data['points']:
                                    row_ids.append(p['customdata'])
                                self.subset = row_ids
                            else:
                                self.subset = [d['_index_'] for d in data]
                        else:
                            self.subset = self.list_index
                    else:
                        self.subset = [d['_index_'] for d in data]
                else:
                    row_ids = []
                    if selected_data is not None and len(selected_data) > 1:
                        # we plot prediction picking subset
                        for p in selected_data['points']:
                            row_ids.append(p['customdata'])
                        self.subset = row_ids
                    else:
                        # we plot filter subset
                        self.subset = [d['_index_'] for d in data]

            self.components['graph']['feature_selector'].figure = \
                self.explainer.plot.contribution_plot(
                    col=self.selected_feature,
                    selection=self.subset,
                    label=self.label,
                    violin_maxf=violin,
                    max_points=points,
                    zoom=zoom_active
                )

            self.components['graph']['feature_selector'].figure['layout'].clickmode = 'event+select'
            # Adjust graph with adding x and y axis titles
            self.components['graph']['feature_selector'].adjust_graph(
                x_ax=truncate_str(self.selected_feature, 110),
                y_ax='Contribution')
            return self.components['graph']['feature_selector'].figure

        @app.callback(
            [
                Output('index_id', 'value'),
                Output("index_id", "n_submit")
            ],
            [
                Input('feature_selector', 'clickData'),
                Input('prediction_picking', 'clickData'),
                Input('dataset', 'active_cell'),
                Input('apply_filter', 'n_clicks'),
                Input('reset_dropdown_button', 'n_clicks'),
                Input({'type': 'del_dropdown_button', 'index': ALL}, 'n_clicks')
            ],
            [
                State('dataset', 'data'),
                State('index_id', 'value')  # Get the current value of the index
            ]
        )
        def update_index_id(click_data,
                            prediction_picking,
                            cell,
                            apply_filters,
                            reset_filter,
                            nclicks_del,
                            data,
                            current_index_id):
            """
            This function is used to update index value according to
            active cell, filters and click data on feature plot or on
            prediction picking graph.
            ----------------------------------------------------------------
            click_data: click on feature selector
            prediction_picking: click on prediction picking graph
            cell: selected sell on dataset
            apply_filters: click on Apply filter button
            reset_filter: click on reset filter button
            nclicks_del: click on del button
            data: dataset
            current_index_id: the current value of the index
            ----------------------------------------------------------------
            return
            selected index id
            boolean n_submit
            """
            ctx = dash.callback_context
            selected = None
            if ctx.triggered[0]['prop_id'] != 'dataset.data':
                if ctx.triggered[0]['prop_id'] == 'feature_selector.clickData':
                    selected = click_data['points'][0]['customdata'][1]
                    self.click_graph = True
                elif ctx.triggered[0]['prop_id'] == 'prediction_picking.clickData':
                    selected = prediction_picking['points'][0]['customdata']
                    self.click_graph = True
                elif ctx.triggered[0]['prop_id'] == 'dataset.active_cell':
                    if cell is not None:
                        selected = data[cell['row']]['_index_']
                    else:
                        # Get actual value in field to refresh the selected value
                        selected = current_index_id
                elif ctx.triggered[0]['prop_id'] == '.':
                    selected = data[0]['_index_']
                # If click on Reset apply button
                elif ctx.triggered[0]['prop_id'] == 'reset_dropdown_button.n_clicks':
                    # Get the row index value
                    selected = data[cell['row']]['_index_']
                # If click on Apply filter button
                elif ctx.triggered[0]['prop_id'] == 'apply_filter.n_clicks':
                    # get the first index on the dataset
                    selected = data[0]['_index_']
                # If click on the last del button
                elif (('del_dropdown_button' in ctx.triggered[0]['prop_id']) &
                      (None not in nclicks_del)):
                    # Get the row index value
                    selected = data[cell['row']]['_index_']
                else:
                    selected = data[0]['_index_']
            else:
                raise PreventUpdate
            return selected, True

        @app.callback(
            Output('threshold_label', 'children'),
            [Input('threshold_id', 'value')])
        def update_threshold_label(value):
            """
            update threshold label
            """
            return f'Threshold: {value}'

        @app.callback(
            Output('max_contrib_label', 'children'),
            [Input('max_contrib_id', 'value')])
        def update_max_contrib_label(value):
            """
            update max_contrib label
            """
            self.components['filter']['max_contrib']['max_contrib_id'].value = value
            return f'Features to display: {value}'

        @app.callback(
            [Output('max_contrib_id', 'value'),
             Output('max_contrib_id', 'max'),
             Output('max_contrib_id', 'marks')
             ],
            [Input('modal', 'is_open')],
            [State('features', 'value')]
        )
        def update_max_contrib_id(is_open,
                                  features):
            """
            update max contrib component layout after settings modifications
            """
            ctx = dash.callback_context
            if ctx.triggered[0]['prop_id'] == 'modal.is_open':
                if is_open:
                    raise PreventUpdate
                else:
                    max = min(features, len(self.dataframe.columns) - 2)
                    if max // 5 == max / 5:
                        nb_marks = min(int(max // 5), 10)
                    elif max // 4 == max / 4:
                        nb_marks = min(int(max // 4), 10)
                    elif max // 3 == max / 3:
                        nb_marks = min(int(max // 3), 10)
                    elif max // 7 == max / 7:
                        nb_marks = min(int(max // 6), 10)
                    else:
                        nb_marks = 2
                    marks = {f'{round(max * feat / nb_marks)}': f'{round(max * feat / nb_marks)}'
                             for feat in range(1, nb_marks + 1)}
                    marks['1'] = '1'
                    if max < self.components['filter']['max_contrib']['max_contrib_id'].value:
                        value = max
                    else:
                        value = no_update

                    return value, max, marks

        @app.callback(
            Output(component_id='detail_feature', component_property='figure'),
            [
                Input('threshold_id', 'value'),
                Input('max_contrib_id', 'value'),
                Input('check_id_positive', 'value'),
                Input('check_id_negative', 'value'),
                Input('masked_contrib_id', 'value'),
                Input('select_label', 'value'),
                Input('dataset', 'active_cell'),
                Input('feature_selector', 'clickData'),
                Input('prediction_picking', 'clickData'),
                Input("validation", "n_clicks"),
                Input('bool_groups', 'on'),
                Input('ember_detail_feature', 'n_clicks'),
            ],
            [
                State('index_id', 'value'),
                State('dataset', 'data')
            ]
        )
        def update_detail_feature(threshold,
                                  max_contrib,
                                  positive,
                                  negative,
                                  masked,
                                  label,
                                  cell,
                                  click_data,
                                  prediction_picking,
                                  validation_click,
                                  bool_group,
                                  click_zoom,
                                  index,
                                  data):
            """
            update local explanation plot according to app changes.
            -------------------------------------------------------
            threshold: threshold
            max_contrib: max contribution
            positive: boolean
            negative: boolean
            masked: feature(s) to mask
            label: label
            cell: selected cell
            click_data: click on feature selector graph
            prediction_picking: click on prediction picking graph
            validation_click: click on validation
            bool_group: boolean
            click_zoom: click on zoom button
            index: selected index
            data: the dataset
            --------------------------------------------------------
            return
            detail feature graph
            """
            # Zoom is False by Default. It becomes True if we click on it
            click = 2 if click_zoom is None else click_zoom
            if click % 2 == 0:
                zoom_active = False
            else:
                zoom_active = True
            ctx = dash.callback_context
            selected = None
            if ctx.triggered[0]['prop_id'] == 'feature_selector.clickData':
                selected = click_data['points'][0]['customdata'][1]
            elif ctx.triggered[0]['prop_id'] == 'prediction_picking.clickData':
                selected = prediction_picking['points'][0]['customdata']
            elif ctx.triggered[0]['prop_id'] in ['threshold_id.value', 'validation.n_clicks']:
                selected = index
            elif ctx.triggered[0]['prop_id'] == 'dataset.active_cell':
                if cell:
                    selected = data[cell['row']]['_index_']
                else:
                    zoom_active = zoom_active
                    # raise PreventUpdate
            else:
                selected = index
            threshold = threshold if threshold != 0 else None
            if positive == [1]:
                sign = (None if negative == [1] else True)
            else:
                sign = (False if negative == [1] else None)
            self.explainer.filter(threshold=threshold,
                                  features_to_hide=masked,
                                  positive=sign,
                                  max_contrib=max_contrib,
                                  display_groups=bool_group)
            if np.issubdtype(type(self.explainer.x_init.index[0]), np.dtype(int).type):
                selected = int(selected)
            self.components['graph']['detail_feature'].figure = self.explainer.plot.local_plot(
                index=selected,
                label=label,
                show_masked=True,
                yaxis_max_label=8,
                display_groups=bool_group,
                zoom=zoom_active
            )
            # Adjust graph with adding x axis titles
            self.components['graph']['detail_feature'].adjust_graph(x_ax='Contribution')
            # font size can be adapted to screen size
            list_yaxis = [self.components['graph']['detail_feature'].figure.data[i].y[0] for i in
                          range(len(self.components['graph']['detail_feature'].figure.data))]
            # exclude new line with labels of y axis
            list_yaxis = [x.split('<br />')[0] for x in list_yaxis]
            nb_car = max([len(x) for x in list_yaxis])
            self.components['graph']['detail_feature'].figure.update_layout(
                yaxis=dict(tickfont={'size': min(round(500 / nb_car), 12)})
            )
            return self.components['graph']['detail_feature'].figure

        @app.callback(
            Output("validation", "n_clicks"),
            [
                Input("index_id", "n_submit")
            ],
        )
        def click_validation(n_submit):
            """
            submit index selection
            """
            if n_submit:
                return 1
            else:
                raise PreventUpdate

        @app.callback(
            [
                Output('dataset', 'style_data_conditional'),
                Output('dataset', 'style_filter_conditional'),
                Output('dataset', 'style_header_conditional'),
                Output('dataset', 'style_cell_conditional'),
            ],
            [
                Input("validation", "n_clicks")
            ],
            [
                State('dataset', 'data'),
                State('index_id', 'value')
            ]
        )
        def datatable_layout(validation,
                             data,
                             index):
            ctx = dash.callback_context
            if ctx.triggered[0]['prop_id'] == 'validation.n_clicks' and validation is not None:
                pass
            else:
                raise PreventUpdate

            style_data_conditional = [
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgb(248, 248, 248)'
                }
            ]
            style_filter_conditional = []
            style_header_conditional = [
                {'if': {'column_id': c}, 'fontWeight': 'bold'}
                for c in ['_index_', '_predict_']
            ]
            style_cell_conditional = [
                {'if': {'column_id': c},
                 'width': '70px', 'fontWeight': 'bold'} for c in ['_index_', '_predict_']
            ]

            selected = check_row(data, index)
            if selected is not None:
                style_data_conditional += [{"if": {"row_index": selected}, "backgroundColor": self.color[0]}]

            return style_data_conditional, style_filter_conditional, style_header_conditional, style_cell_conditional

        @app.callback(
            Output(component_id='prediction_picking', component_property='figure'),
            [
                Input('global_feature_importance', 'clickData'),
                Input('dataset', 'data'),
                Input('apply_filter', 'n_clicks'),
                Input('reset_dropdown_button', 'n_clicks'),
                Input({'type': 'del_dropdown_button', 'index': ALL}, 'n_clicks'),
                Input('select_label', 'value'),
                Input('modal', 'is_open'),
                Input('ember_prediction_picking', 'n_clicks')
            ],
            [
                State('points', 'value'),
                State('violin', 'value')
            ]
        )
        def update_prediction_picking(feature,
                                      data,
                                      apply_filters,
                                      reset_filter,
                                      nclicks_del,
                                      label,
                                      is_open,
                                      click_zoom,
                                      points,
                                      violin):
            """
            Update feature plot according to label, data,
            selected feature and settings modifications
            ------------------------------------------------
            feature: click on features importance graph
            data: the dataset
            apply_filters: click on apply filter button
            reset_filter: click on reset filter button
            nclicks_del: click on del button
            label: selected label
            is_open: modal
            click_zoom: click on zoom button
            points: number of points
            violin: number of violin plot
            -------------------------------------------------
            return
            prediction picking graph
            """
            ctx = dash.callback_context
            # Filter subset
            filter_subset = None
            if not ctx.triggered:
                raise dash.exceptions.PreventUpdate
            if ctx.triggered[0]['prop_id'] == 'modal.is_open':
                if is_open:
                    raise PreventUpdate
                else:
                    self.settings['points'] = points
                    self.settings_ini['points'] = self.settings['points']
                    self.settings['violin'] = violin
                    self.settings_ini['violin'] = self.settings['violin']
            elif ctx.triggered[0]['prop_id'] == 'select_label.value':
                self.label = label
                self.subset = None
            # If we have clicked on reset button
            elif ctx.triggered[0]['prop_id'] == 'reset_dropdown_button.n_clicks':
                self.subset = None
            # If we have clicked on Apply filter button
            elif ctx.triggered[0]['prop_id'] == 'apply_filter.n_clicks':
                self.subset = [d['_index_'] for d in data]
            # If we have clicked on the last delete button (X)
            elif (('del_dropdown_button' in ctx.triggered[0]['prop_id']) &
                  (None not in nclicks_del)):
                self.subset = None
            else:
                raise PreventUpdate

            if self.explainer.y_target is not None:
                self.components['graph']['prediction_picking'].figure = self.explainer.plot.scatter_plot_prediction(
                    selection=self.subset,
                    max_points=points,
                    label=self.label
                )

                self.components['graph']['prediction_picking'].figure['layout'].clickmode = 'event+select'
                # Adjust graph with adding x and y axis titles
                self.components['graph']['prediction_picking'].adjust_graph(
                    x_ax="True Values",
                    y_ax="Predicted Values")
            else:
                fig = go.Figure()
                fig.update_layout(
                xaxis =  { "visible": False },
                yaxis = { "visible": False },
                annotations = [
                    {
                        "text": "Provide the y_target argument in the compile() method to display this plot.",
                        "xref": "paper",
                        "yref": "paper",
                        "showarrow": False,
                        "font": {
                            "size": 14
                        }
                    }
                ]
            )
                self.components['graph']['prediction_picking'].figure = fig

            return self.components['graph']['prediction_picking'].figure

        @app.callback(
            Output("modal_feature_importance", "is_open"),
            [Input("open_feature_importance", "n_clicks"),
             Input("close_feature_importance", "n_clicks")],
            [State("modal_feature_importance", "is_open")],
        )
        def toggle_modal_feature_importancet(n1,
                                             n2,
                                             is_open):
            """
            Function used to open and close modal explication when we click
            on "?" button on feature_importance graph
            ---------------------------------------------------------------
            n1: click on "?" button
            n2: click on close button in modal
            ---------------------------------------------------------------
            return modal
            """
            if n1 or n2:
                return not is_open
            return is_open

        @app.callback(
            Output("modal_feature_selector", "is_open"),
            [Input("open_feature_selector", "n_clicks"),
             Input("close_feature_selector", "n_clicks")],
            [State("modal_feature_selector", "is_open")],
        )
        def toggle_modal_feature_selector(n1,
                                          n2,
                                          is_open):
            """
            Function used to open and close modal explication when we click
            on "?" button on feature_selector graph
            ---------------------------------------------------------------
            n1: click on "?" button
            n2: click on close button in modal
            ---------------------------------------------------------------
            return modal
            """
            if n1 or n2:
                return not is_open
            return is_open

        @app.callback(
            Output("modal_detail_feature", "is_open"),
            [Input("open_detail_feature", "n_clicks"),
             Input("close_detail_feature", "n_clicks")],
            [State("modal_detail_feature", "is_open")],
        )
        def toggle_modal_detail_feature(n1,
                                        n2,
                                        is_open):
            """
            Function used to open and close modal explication when we click
            on "?" button on detail_feature graph
            ---------------------------------------------------------------
            n1: click on "?" button
            n2: click on close button in modal
            ---------------------------------------------------------------
            return modal
            """
            if n1 or n2:
                return not is_open
            return is_open

        @app.callback(
            Output("modal_prediction_picking", "is_open"),
            [Input("open_prediction_picking", "n_clicks"),
             Input("close_prediction_picking", "n_clicks")],
            [State("modal_prediction_picking", "is_open")],
        )
        def toggle_modal_prediction_picking(n1,
                                            n2,
                                            is_open):
            """
            Function used to open and close modal explication when we click
            on "?" button on prediction_picking graph
            ---------------------------------------------------------------
            n1: click on "?" button
            n2: click on close button in modal
            ---------------------------------------------------------------
            return modal
            """
            if n1 or n2:
                return not is_open
            return is_open

        @app.callback(
            Output("modal_filter", "is_open"),
            [Input("open_filter", "n_clicks"),
             Input("close_filter", "n_clicks")],
            [State("modal_filter", "is_open")],
        )
        def toggle_modal_filters(n1,
                                 n2,
                                 is_open):
            """
            Function used to open and close modal explication when we click
            on "?" button on Dataset Filters Tab
            ---------------------------------------------------------------
            n1: click on "?" button
            n2: click on close button in modal
            ---------------------------------------------------------------
            return modal
            """
            if n1 or n2:
                return not is_open
            return is_open

        # Add or remove plot blocs in the 'dropdowns_container'
        @app.callback(
            Output('dropdowns_container', 'children'),
            [
                Input('add_dropdown_button', 'n_clicks'),
                Input('reset_dropdown_button', 'n_clicks'),
                Input({'type': 'del_dropdown_button', 'index': ALL}, 'n_clicks')
            ],
            [
                 State('dropdowns_container', 'children'),
                 State('name', 'value')
            ]
        )
        def layout_filter(n_clicks_add,
                          n_clicks_rm,
                          n_clicks_reset,
                          currents_filters,
                          name
                          ):
            """
            Function used to create filter blocs in the dropdowns_container.
            Each bloc will contains:
                -label
                -dropdown button to select feature to filter
                -div which will contains modalities
                -delete button
            ---------------------------------------------------------------
            n_clicks_add: click on add filter
            n_clicks_reset: click on reset filter button
            n_click_del: click on delete button
            children: information on dropdown container
            name: name for feature name
            ---------------------------------------------------------------
            return
                filter blocs
            """
            # Context and init handling (no action)
            ctx = dash.callback_context
            if not ctx.triggered:
                raise dash.exceptions.PreventUpdate
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]

            # We use domain name for feature name
            dict_name = [self.explainer.features_dict[i]
                         for i in self.dataframe.drop(['_index_', '_predict_'], axis=1).columns]
            dict_id = [i for i in self.dataframe.drop(['_index_', '_predict_'], axis=1).columns]
            # Create dataframe to sort it by feature_name
            df_feature_name = pd.DataFrame({'feature_name': dict_name,
                                            'feature_id': dict_id})
            df_feature_name = df_feature_name.sort_values(
                by='feature_name').reset_index(drop=True)
            # Options are sorted by feature_name
            options = [
                {"label": '_index_', "value": '_index_'},
                {"label": '_predict_', "value": '_predict_'}] + \
                [{"label": df_feature_name.loc[i, 'feature_name'],
                  "value": df_feature_name.loc[i, 'feature_id']}
                 for i in range(len(df_feature_name))]

            # Creation of a new graph
            if button_id == 'add_dropdown_button':
                # ID index definition
                if n_clicks_add is None:
                    index_id = 0
                else:
                    index_id = n_clicks_add
                # Appending a dropdown block to 'dropdowns_container'children
                subset_filter = html.Div(
                    id={'type': 'bloc_div',
                        'index': index_id},
                    children=[
                        html.Div([
                            html.Br(),
                            # div which will contains label
                            html.Div(
                                    id={'type': 'dynamic-output-label',
                                        'index': index_id},
                                    )
                            ]),
                        html.Div([
                            # div with dopdown button to select feature to filter
                            html.Div(dcc.Dropdown(
                                id={'type': 'var_dropdown',
                                    'index': index_id},
                                options=options,
                                placeholder="Variable"
                            ), style={"width": "30%"}),
                            # div which will contains modalities
                            html.Div(
                                 id={'type': 'dynamic-output',
                                     'index': index_id},
                                 style={"width": "50%"}
                                ),
                            # Button to delete bloc
                            dbc.Button(
                                id={'type': 'del_dropdown_button',
                                    'index': index_id},
                                children='X',
                                color='warning',
                                size='sm'
                            )
                        ], style={'display': 'flex'})
                    ]
                )
                return currents_filters + [subset_filter]
            # Removal of all existing filters
            elif button_id == 'reset_dropdown_button':
                return [html.Div(
                    id={'type': 'bloc_div',
                        'index': 0},
                    children=[])]
            # Removal of an existing filter
            else:
                filter_id_to_remove = eval(button_id)['index']
                return [gr for gr in currents_filters
                        if gr['props']['id']['index'] != filter_id_to_remove]

        @app.callback(
            Output('reset_dropdown_button', 'disabled'),
            [Input('add_dropdown_button', 'n_clicks'),
             Input('reset_dropdown_button', 'n_clicks'),
             Input({'type': 'del_dropdown_button', 'index': ALL}, 'n_clicks')]
        )
        def update_disabled_reset_button(n_click_add,
                                         n_click_reset,
                                         n_click_del):
            """
            Function used to disabled or not the reset filter button.
            This button is disabled if there is no filter added.
            ---------------------------------------------------------------
            n_click_add: click on add filter button
            n_click_reset: click on reset filter button
            n_click_del: click on delete button
            ---------------------------------------------------------------
            return disabled style
            """
            ctx = dash.callback_context
            if ctx.triggered[0]['prop_id'] == 'add_dropdown_button.n_clicks':
                disabled = False
            elif ctx.triggered[0]['prop_id'] == 'reset_dropdown_button.n_clicks':
                disabled = True
            elif None not in n_click_del:
                disabled = True
            else:
                disabled = False
            return disabled

        @app.callback(
            Output('apply_filter', 'style'),
            [Input('add_dropdown_button', 'n_clicks'),
             Input('reset_dropdown_button', 'n_clicks'),
             Input({'type': 'del_dropdown_button', 'index': ALL}, 'n_clicks')]
        )
        def update_style_apply_filter_button(n_click_add,
                                             n_click_reset,
                                             n_click_del):
            """
            Function used to display or not the apply filter button.
            This button is only display if almost one filter was added.
            ---------------------------------------------------------------
            n_click_add: click on add filter button
            n_click_reset: click on reset filter button
            n_click_del: click on delete button
            ---------------------------------------------------------------
            return style of apply filter button
            """
            ctx = dash.callback_context
            if ctx.triggered[0]['prop_id'] == 'add_dropdown_button.n_clicks':
                return {'display': 'block'}
            elif ctx.triggered[0]['prop_id'] == 'reset_dropdown_button.n_clicks':
                return {'display': 'none'}
            elif None not in n_click_del:
                return {'display': 'none'}
            else:
                return {'display': 'block'}

        @app.callback(
            Output({'type': 'dynamic-output-label', 'index': MATCH}, 'children'),
            Input({'type': 'var_dropdown', 'index': MATCH}, 'value'),
        )
        def update_label_filter(value):
            """
            Function used to add label to the filters. Label is updated
            when value is not None
            ---------------------------------------------------------------
            value: value selected on the var dropdown button
            ---------------------------------------------------------------
            return label
            """
            if value is not None:
                return html.Label("Variable {} is filtered".format(value))
            else:
                return html.Label('Select variable to filter')

        @app.callback(
            Output({'type': 'dynamic-output', 'index': MATCH}, 'children'),
            [Input({'type': 'var_dropdown', 'index': MATCH}, 'value'),
             Input({'type': 'var_dropdown', 'index': MATCH}, 'id'),
             Input('add_dropdown_button', 'n_clicks')],
        )
        def display_output(value,
                           id,
                           add_click):
            """
            Function used to create modalities choices. Componenents are different
            according to the type of the selected variable.
            For string variable: component is a dropdown button
            For boolean variable: component is a RadioItems button
            For Integer variable that have less than 20 modalities: component
            is a dropdown button.
            For date variable: component is a DatePickerRange
            Else: components are lower and upper values
            ---------------------------------------------------------------
            value: value selected on the var dropdown button
            id: id of the var dropdown button
            add_click: click on add_dropdown_button
            ---------------------------------------------------------------
            return modalities components. If the component is new, value
            is empty by default.
            """
            # Context and init handling (no action)
            ctx = dash.callback_context
            if not ctx.triggered:
                raise dash.exceptions.PreventUpdate
            # No update last modalities values if we click on add button
            if ctx.triggered[0]['prop_id'] == 'add_dropdown_button.n_clicks':
                raise dash.exceptions.PreventUpdate
            # Creation on modalities dropdown button
            else:
                if value is not None:
                    if type(self.round_dataframe[value].iloc[0]) == bool:
                        new_element = html.Div(dcc.RadioItems(
                            [{'label': val, 'value': val} for
                             val in self.round_dataframe[value].unique()],
                            id={'type': 'dynamic-bool',
                                'index': id['index']},
                            value=self.round_dataframe[value].iloc[0],
                            inline=False
                            ), style={"width": "65%", 'margin-left': '20px'})
                    elif (type(self.round_dataframe[value].iloc[0]) == str) | \
                         ((type(self.round_dataframe[value].iloc[0]) == np.int64) &
                          (len(self.round_dataframe[value].unique()) <= 20)):
                        new_element = html.Div(dcc.Dropdown(
                            id={
                               'type': 'dynamic-str',
                               'index': id['index']
                            },
                            options=[{'label': i, 'value': i} for
                                    i in np.sort(self.round_dataframe[value].unique())],
                            multi=True,
                            ), style={"width": "65%", 'margin-left': '20px'})
                    elif ((type(self.round_dataframe[value].iloc[0]) is pd.Timestamp) |
                          (type(self.round_dataframe[value].iloc[0]) is datetime.datetime)):
                        new_element = html.Div(
                            dcc.DatePickerRange(
                                id={
                                   'type': 'dynamic-date',
                                   'index': id['index']
                                },
                                min_date_allowed=self.round_dataframe[value].min(),
                                max_date_allowed=self.round_dataframe[value].max(),
                                start_date=self.round_dataframe[value].min(),
                                end_date=self.round_dataframe[value].max()
                               ), style={'width': '65%', 'margin-left': '20px'}),
                    else:
                        lower_value = 0
                        upper_value = 0
                        new_element = html.Div([
                                            dcc.Input(
                                                id={
                                                    'type': 'lower',
                                                    'index': id['index']
                                                },
                                                value=lower_value,
                                                type="number",
                                                style={'width': '60px'}),
                                            ' <= {} in [{}, {}]<= '.format(
                                                value,
                                                self.round_dataframe[value].min(),
                                                self.round_dataframe[value].max()),
                                            dcc.Input(
                                                id={
                                                    'type': 'upper',
                                                    'index': id['index']
                                                },
                                                value=upper_value,
                                                type="number",
                                                style={'width': '60px'}
                                            )
                        ], style={'margin-left': '20px'})
                else:
                    new_element = html.Div()
                return new_element
