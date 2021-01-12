"""
Main class of Web application Shapash
"""
import dash
import dash_table
from dash import no_update
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Output, Input, State
from flask import Flask
import pandas as pd
import plotly.graph_objs as go
import random
import numpy as np
import re
from math import log10
from shapash.webapp.utils.utils import apply_filter, check_row, round_to_1
from shapash.webapp.utils.MyGraph import MyGraph


def create_input_modal(id, label, tooltip):
    return dbc.FormGroup(
        [
            dbc.Label(label, id=f'{id}_label', html_for=id, width=8),
            dbc.Col(
                dbc.Input(id=id, type="number", value=0),
                width=4),
            dbc.Tooltip(tooltip, target=f'{id}_label', placement='bottom'),
        ], row=True,
    )


class SmartApp:
    """
        Bridge pattern decoupling the application part from SmartExplainer and SmartPlotter.
        Attributes
        ----------
        explainer: object
            SmartExplainer instance to point to.
    """

    def __init__(self, explainer):
        """
        Init on class instantiation, everything to be able to run the app on server.
        Parameters
        ----------
        explainer : SmartExplainer
            SmartExplainer object
        """
        # APP
        self.server = Flask(__name__)
        self.app = dash.Dash(
            server=self.server,
            external_stylesheets=[dbc.themes.BOOTSTRAP],
        )
        self.app.title = 'Shapash Monitor'
        self.explainer = explainer

        # SETTINGS
        self.logo = self.app.get_asset_url('shapash-fond-fonce.png')
        self.color = '#f4c000'
        self.bkg_color = "#343736"
        self.settings_ini = {
            'rows': 1000,
            'points': 1000,
            'violin': 10,
            'features': 20,
        }
        self.settings = self.settings_ini.copy()
        self.predict_col = ['_predict_']
        self.explainer.features_imp = self.explainer.state.compute_features_import(self.explainer.contributions)
        if self.explainer._case == 'classification':
            self.label = self.explainer.check_label_name(len(self.explainer._classes) - 1, 'num')[1]
            self.selected_feature = self.explainer.features_imp[-1].idxmax()
            self.max_threshold = int(max([x.applymap(lambda x: round_to_1(x)).max().max()
                                          for x in self.explainer.contributions]))
        else:
            self.label = None
            self.selected_feature = self.explainer.features_imp.idxmax()
            self.max_threshold = int(self.explainer.contributions.applymap(lambda x: round_to_1(x)).max().max())
        self.list_index = []
        self.subset = None

        # DATA
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
            self.dataframe = self.explainer.x_pred.copy()
            if isinstance(self.explainer.y_pred, (pd.Series, pd.DataFrame)):
                self.predict_col = self.explainer.y_pred.columns.to_list()[0]
                self.dataframe = self.dataframe.join(self.explainer.y_pred)
            elif isinstance(self.explainer.y_pred, list):
                self.dataframe = self.dataframe.join(pd.DataFrame(data=self.explainer.y_pred,
                                                                  columns=[self.predict_col],
                                                                  index=self.explainer.x_pred.index))
            else:
                raise TypeError('y_pred must be of type pd.Series, pd.DataFrame or list')
        else:
            raise ValueError('y_pred must be set when calling compile function.')

        self.dataframe['_index_'] = self.explainer.x_pred.index
        self.dataframe.rename(columns={f'{self.predict_col}': '_predict_'}, inplace=True)
        col_order = ['_index_', '_predict_'] + self.dataframe.columns.drop(['_index_', '_predict_']).tolist()
        self.list_index = random.sample(population=self.dataframe.index.tolist(),
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
                    self.round_dataframe[col] = self.dataframe[col].map(f'{{:.{digit}f}}'.format)

    def init_components(self):
        """
        Initialize components (graph, table, filter, settings, ...) and insert it inside
        components containers which are created by init_skeleton
        """

        self.components['settings']['input_rows'] = create_input_modal(
            id='rows',
            label="Number of rows for subset",
            tooltip="Set max number of lines for subset (datatable).Filter will be apply on this subset."
        )

        self.components['settings']['input_points'] = create_input_modal(
            id='points',
            label="Number of points for plot",
            tooltip="Set max number of points in feature contribution plots"
        )

        self.components['settings']['input_features'] = create_input_modal(
            id='features',
            label="Number of features to plot",
            tooltip="Set max number of features to plot in features importance and local explanation plots.",
        )

        self.components['settings']['input_violin'] = create_input_modal(
            id='violin',
            label="Max number of labels for violin plot",
            tooltip="Set max number of labels to display a violin plot for feature contribution plot (otherwise a "
                    "scatter plot is displayed)."
        )

        self.components['settings']['name'] = dbc.FormGroup(
            [
                dbc.Checklist(
                    options=[{"label": "Use domain name for features name.", "value": 1}], value=[], inline=True,
                    id="name",
                    style={"margin-left": "20px"}
                ),
                dbc.Tooltip("Replace technical feature names by domain names if exists.",
                            target='name', placement='bottom'),
            ], row=True,
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
                        html.H4(
                            [dbc.Badge("Regression", id='regression_badge', style={"margin-right": "5px"}),
                             dbc.Badge("Classification", id='classification_badge')
                             ],
                            style={"margin": "0px"}
                        ),
                    ],
                    width="auto", align="center",
                ),
                dbc.Col(
                    dbc.Collapse(
                        dbc.FormGroup(
                            [
                                dbc.Label("Class to analyse", style={'color': 'white', 'margin': '0px 5px'}),
                                dcc.Dropdown(
                                    id="select_label",
                                    options=[], value=None,
                                    clearable=False, searchable=False,
                                    style={"verticalAlign": "middle", "zIndex": '1010', "min-width": '200px'}
                                )
                            ],
                            row=True, style={"margin": "0px 0px 0px 5px", "align-items": "center"}
                        ),
                        is_open=True, id='select_collapse'
                    ),
                    width="auto", align="center", style={'padding': 'none'}
                ),
                dbc.Col(
                    html.Div(
                        [
                            html.Img(id='settings', title='settings', alt='Settings',
                                     src=self.app.get_asset_url('settings.png'),
                                     height='40px',
                                     style={'cursor': 'pointer'}),
                            self.components['settings']['modal'],
                        ]
                    ),
                    align="center", width="50px", style={'padding': '0px 0px 0px 20px'}
                )
            ],
            form=True, no_gutters=True, justify="end"
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
                    [{"name": i, "id": i} for i in self.explainer.x_pred],
            editable=False, row_deletable=False,
            style_as_list_view=True,
            virtualization=True,
            page_action='none',
            fixed_rows={'headers': True, 'data': 0},
            fixed_columns={'headers': True, 'data': 0},
            filter_action='custom', filter_query='',
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

        self.components['graph']['detail_feature'] = MyGraph(
            figure=go.Figure(), id='detail_feature'
        )

        self.components['filter']['index'] = dbc.FormGroup(
            [
                dbc.Label("Index ", align="center", width=4),
                dbc.Col(
                    dbc.Input(
                        id="index_id", type="text", bs_size="md", placeholder="Id must exist",
                        debounce=True, persistence=True, style={'textAlign': 'right'}
                    ),
                    style={'padding': "0px"}
                ),
                dbc.Col(
                    html.Img(id='validation', alt='Validate', title='Validate index',
                             src=self.app.get_asset_url('reload.png'),
                             height='30px', style={'cursor': 'pointer'},
                             ),
                    style={'padding': "0px"}, align="center", width="40px"
                )
            ],
            row=True
        )

        self.components['filter']['threshold'] = dbc.FormGroup(
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

        self.components['filter']['max_contrib'] = dbc.FormGroup(
            [
                dbc.Label(
                    "Features to display : ", id='max_contrib_label'),
                dcc.Slider(
                    min=1, max=min(self.settings['features'], len(self.dataframe.columns) - 2),
                    step=1, value=min(self.settings['features'], len(self.dataframe.columns) - 2),
                    id="max_contrib_id",
                )
            ],
            className='filter_dashed'
        )

        self.components['filter']['positive_contrib'] = dbc.FormGroup(
            [
                dbc.Label("Contributions to display : "),
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.Checklist(
                                options=[{"label": "Positive", "value": 1}], value=[1], inline=True,
                                id="check_id_positive"
                            ), width=6
                        ),
                        dbc.Col(
                            dbc.Checklist(
                                options=[{"label": "Negative", "value": 1}], value=[1], inline=True,
                                id="check_id_negative"
                            ), width=6, style={'padding': "0px"}, align="center"
                        ),
                    ], no_gutters=True, justify="center", form=True
                )
            ],
            className='filter_dashed'
        )

        self.components['filter']['masked_contrib'] = dbc.FormGroup(
            [
                dbc.Label(
                    "Feature(s) to mask :"),
                dcc.Dropdown(
                    options=[{'label': key, 'value': value} for key, value in self.explainer.inv_features_dict.items()],
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
                dbc.Row(
                    [
                        dbc.Col(
                            html.A(
                                dbc.Row(
                                    [
                                        html.Img(src=self.logo, height="40px"),
                                        html.H4("Shapash Monitor", id="shapash_title"),
                                    ],
                                    align="center",
                                ),
                                href="https://github.com/MAIF/shapash", target="_blank",
                            ),
                            md=3, align="left"
                        ),
                        dbc.Col(
                            self.components['menu'],
                            md=9, align='right',
                        )
                    ],
                    style={'padding': "5px 15px", "verticalAlign": "middle"},
                )
            ],
            fluid=True, style={'height': '50px', 'backgroundColor': self.bkg_color},
        )

        self.skeleton['body'] = dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Div(
                                    self.draw_component('graph', 'global_feature_importance'),
                                    className="card",
                                    id="card_global_feature_importance",
                                )
                            ],
                            md=5,
                            align="center",
                            style={'padding': '0px 10px'},
                        ),
                        dbc.Col(
                            [
                                html.Div(
                                    self.draw_component('table', 'dataset'),
                                    className="card",
                                    id='card_dataset',
                                    style={'cursor': 'pointer'},
                                )
                            ],
                            md=7,
                            align="center",
                            style={'padding': '0px 10px'},
                        ),
                    ],
                    style={'padding': '15px 10px 0px 10px'},
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Div(
                                    self.draw_component('graph', 'feature_selector'),
                                    className="card",
                                    id='card_feature_selector',
                                )
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
                                                html.Div(
                                                    self.draw_component('graph', 'detail_feature'),
                                                    className="card",
                                                    id='card_detail_feature',
                                                ),
                                            ],
                                            md=8,
                                            align="center",
                                            # style={'height': '27rem'}
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
                    style={'padding': '15px 10px'},
                ),
            ],
            className="mt-12",
            fluid=True
        )

    def adjust_menu(self):
        """
        Override menu from explainer object depending on classification or regression case.
        """
        on_style = {'backgroundColor': self.color, 'color': self.bkg_color, 'margin-right': '0.5rem'}
        off_style = {'backgroundColor': '#71653B', 'color': self.bkg_color, 'margin-right': '0.5rem'}
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

    def draw_component(self, component_type, component_id, title=None):
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
            (combining for example Graph + embed button to get fullscreen details)
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
                **{'data-component-type': component_type}
            )
        )
        return component

    def draw_filter(self):
        """
        Method which returns filter components block for local contributions plot.
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

    def select_point(self, graph, click_data):
        """
        Method which set the selected point in graph component corresponding to click_data
        """
        if click_data:
            curve_id = click_data['points'][0]['curveNumber']
            point_id = click_data['points'][0]['pointIndex']
            for curve in range(len(self.components['graph'][graph].figure['data'])):
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
                    Input(f'ember_{component_id}', 'data-component-type')
                ]
            )
            def ember(click, data_component_type):
                click = 2 if click is None else click
                toggle_on = True if click % 2 == 0 else False
                if toggle_on:
                    style_component = {
                        'height': '25rem'
                    }
                    if data_component_type == 'table':
                        style_component = {
                            'maxHeight': '25rem',
                        }
                    this_style_card = {
                        'height': '26rem', 'zIndex': 900,
                    }

                    return this_style_card, style_component

                else:
                    this_style_card = {
                        'height': 'auto', 'width': 'auto',
                        'zIndex': 998,
                        'position': 'fixed', 'top': '50px',
                        'bottom': 0, 'left': 0, 'right': 0,
                    }
                    style_component = {
                        'height': '87vh', 'maxHeight': '87vh',
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
        def toggle_modal(n1, n2, rows, points, features, violin):
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
                Output('dataset', 'columns'),
                Output('dataset', 'active_cell'),
            ],
            [
                Input('dataset', 'sort_by'),
                Input('dataset', "filter_query"),
                Input('modal', 'is_open')
            ],
            [State('rows', 'value'),
             State('name', 'value')]
        )
        def update_datatable(sort_by, filter_query, is_open, rows, name):
            """
            update datatable according to sorting, filtering and settings modifications
            """
            ctx = dash.callback_context
            active_cell = no_update
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
                                  [{"name": self.explainer.features_dict[i], "id": i} for i in self.explainer.x_pred]

            if not filter_query:
                df = self.round_dataframe
            else:
                df = apply_filter(self.round_dataframe, filter_query)

            if len(sort_by):
                df = df.sort_values(
                    [col['column_id'] for col in sort_by],
                    ascending=[
                        col['direction'] == 'asc'
                        for col in sort_by
                    ],
                    inplace=False
                )
            self.components['table']['dataset'].data = df.to_dict('records')

            return self.components['table']['dataset'].data, columns, active_cell

        @app.callback(
            [
                Output('global_feature_importance', 'figure'),
                Output('global_feature_importance', 'clickData')
            ],
            [
                Input('select_label', 'value'),
                Input('dataset', 'data'),
                Input('modal', 'is_open')
            ],
            [State('global_feature_importance', 'clickData'),
             State('dataset', "filter_query"),
             State('features', 'value')]
        )
        def update_feature_importance(label, data, is_open, clickData, filter_query, features):
            """
            update feature importance plot according to selected label and dataset state.
            """
            ctx = dash.callback_context
            if ctx.triggered[0]['prop_id'] == 'modal.is_open':
                
                if is_open :
                    raise PreventUpdate
                else:
                    
                    self.settings['features'] = features
                    self.settings_ini['features'] = self.settings['features']
                    
            elif ctx.triggered[0]['prop_id'] == 'select_label.value':
                self.label = label
            elif ctx.triggered[0]['prop_id'] == 'dataset.data':
                self.list_index = [d['_index_'] for d in data]
            else:
                raise PreventUpdate

            selection = self.list_index if filter_query else None
            self.components['graph']['global_feature_importance'].figure = \
                self.explainer.plot.features_importance(max_features=features,
                                                        selection=selection,
                                                        label=self.label
                                                        )
            self.components['graph']['global_feature_importance'].adjust_graph()
            self.components['graph']['global_feature_importance'].figure.layout.clickmode = 'event+select'
            self.select_point('global_feature_importance', clickData)

            # font size can be adapted to screen size
            nb_car = max([len(self.components['graph']['global_feature_importance'].figure.data[0].y[i]) for i in
                          range(len(self.components['graph']['global_feature_importance'].figure.data[0].y))])
            self.components['graph']['global_feature_importance'].figure.update_layout(
                yaxis=dict(tickfont={'size': min(round(500 / nb_car), 12)})
            )

            return self.components['graph']['global_feature_importance'].figure, clickData

        @app.callback(
            Output(component_id='feature_selector', component_property='figure'),
            [
                Input('global_feature_importance', 'clickData'),
                Input('select_label', 'value'),
                Input('modal', 'is_open')
            ],
            [
                State('points', 'value'),
                State('violin', 'value')
            ]
        )
        def update_feature_selector(feature, label, is_open, points, violin):
            """
            Update feature plot according to label, data, selected feature and settings modifications
            """
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
                    self.selected_feature = feature['points'][0]['label']
                    if feature['points'][0]['curveNumber'] == 0 and \
                            len(self.components['graph']['global_feature_importance'].figure['data']) == 2:
                        self.subset = self.list_index
                    else:
                        self.subset = None
            else:
                raise PreventUpdate

            self.components['graph']['feature_selector'].figure = self.explainer.plot.contribution_plot(
                col=self.selected_feature,
                selection=self.subset,
                label=self.label,
                violin_maxf=violin,
                max_points=points
            )

            self.components['graph']['feature_selector'].figure['layout'].clickmode = 'event'
            subset_graph = True if self.subset is not None else False
            self.components['graph']['feature_selector'].adjust_graph(subset_graph=subset_graph, title_size_adjust=True)

            return self.components['graph']['feature_selector'].figure

        @app.callback(
            [
                Output('index_id', 'value'),
                Output("index_id", "n_submit")
            ],
            [
                Input('feature_selector', 'clickData'),
                Input('dataset', 'active_cell')
            ],
            [
                State('dataset', 'data'),
                State('index_id', 'value') # Get the current value of the index
            ]
        )
        def update_index_id(click_data, cell, data, current_index_id):
            """
            update index value according to active cell and click data on feature plot.
            """
            ctx = dash.callback_context
            if ctx.triggered[0]['prop_id'] != 'dataset.data' :
                if ctx.triggered[0]['prop_id'] == 'feature_selector.clickData':
                    selected = click_data['points'][0]['customdata']
                    self.click_graph = True
                elif ctx.triggered[0]['prop_id'] == 'dataset.active_cell':
                    if cell is not None:
                        selected = data[cell['row']]['_index_']
                    else:
                        selected = current_index_id # Get actual value in field to refresh the selected value
                elif ctx.triggered[0]['prop_id'] == '.' :
                    selected = data[0]['_index_']
            else :
                raise PreventUpdate  
            return selected, True

        @app.callback(
            Output('threshold_label', 'children'),
            [Input('threshold_id', 'value')])
        def update_threshold_label(value):
            """
            update threshold label
            """
            return f'Threshold : {value}'

        @app.callback(
            Output('max_contrib_label', 'children'),
            [Input('max_contrib_id', 'value')])
        def update_max_contrib_label(value):
            """
            update max_contrib label
            """
            self.components['filter']['max_contrib']['max_contrib_id'].value = value
            return f'Features to display : {value}'

        @app.callback(
            [Output('max_contrib_id', 'value'),
             Output('max_contrib_id', 'max'),
             Output('max_contrib_id', 'marks')
             ],
            [Input('modal', 'is_open')],
            [State('features', 'value')]
        )
        def update_max_contrib_id(is_open, features):
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
            ],
            [
                State('index_id', 'value'),
                State('dataset', 'data')
            ]
        )
        def update_detail_feature(threshold, max_contrib, positive, negative, masked, label, cell,
                                  click_data, index, data):
            """
            update local explanation plot according to app changes.
            """
            ctx = dash.callback_context
            selected = None
            if ctx.triggered[0]['prop_id'] == 'feature_selector.clickData':
                selected = click_data['points'][0]['customdata']
            elif ctx.triggered[0]['prop_id'] == 'threshold_id.value':
                selected = index
            elif ctx.triggered[0]['prop_id'] == 'dataset.active_cell':
                if cell:
                    selected = data[cell['row']]['_index_']
                else:
                    raise PreventUpdate

            if selected is None:
                if cell is not None:
                    selected = data[cell['row']]['_index_']
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
                                  max_contrib=max_contrib)
            if np.issubdtype(type(self.explainer.x_pred.index[0]), np.dtype(int).type):
                selected = int(selected)
            self.components['graph']['detail_feature'].figure = self.explainer.plot.local_plot(index=selected,
                                                                                               label=label,
                                                                                               show_masked=True,
                                                                                               yaxis_max_label=0)
            self.components['graph']['detail_feature'].adjust_graph(title_size_adjust=True)
            # font size can be adapted to screen size
            nb_car = max([len(self.components['graph']['detail_feature'].figure.data[i].y[0]) for i in
                          range(len(self.components['graph']['detail_feature'].figure.data))])
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
        def datatable_layout(validation, data, index):
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
                style_data_conditional += [{"if": {"row_index": selected}, "backgroundColor": self.color}]

            return style_data_conditional, style_filter_conditional, style_header_conditional, style_cell_conditional
