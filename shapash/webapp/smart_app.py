"""
Main class of Web application Shapash
"""
import dash
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import dash_daq as daq
from dash.dependencies import Output, Input, State
import dash_table
import pandas as pd
import plotly.graph_objs as go
import random


class MyGraph(dcc.Graph):
    def __init__(self, figure, style, id):
        super().__init__()
        self.figure = figure
        self.style = style
        self.id = id

    def adjust_graph(self):
        """
        Override graph layout for app use
        """
        self.figure.update_layout(
            autosize=True,
            margin=dict(
                l=50,
                r=10,
                b=0,
                t=50,
                pad=0
            ),
            title={
                'y': 1,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'}
        )
        self.figure.update_xaxes(title='', automargin=True)
        self.figure.update_yaxes(title='', automargin=True)


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
        Init on class instanciation, everything to be able to run the app
        on server.
        Parameters
        ----------
        explainer : [type]
            [description]
        """
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.BOOTSTRAP],
        )
        self.app.title = 'Shapash Monitor'
        self.logo = self.app.get_asset_url('shapash-fond-fonce.png')
        self.explainer = explainer

        self.color = '#f4c000'
        self.bkg_color = "#343736"
        # TODO : add constants in front
        self.max_rows = 1000
        self.max_features = 20
        self.violin_maxf = 10
        self.max_points = 2000
        self.page_size = 20

        self.explainer.features_imp = self.explainer.state.compute_features_import(self.explainer.contributions)
        if explainer._case == 'classification':
            self.label = self.explainer.check_label_name(len(self.explainer._classes) - 1, 'num')[1]
            self.selected_feature = explainer.features_imp[-1].idxmax()
            self.max_threshold = int(round(max([abs(x).max().max() for x in self.explainer.contributions])))
        else:
            self.label = None
            self.selected_feature = self.explainer.features_imp.idxmax()
            self.max_threshold = int(round(max(abs(self.explainer.contributions).max())))
        self.max_features = min(20, len(self.explainer.features_desc))
        self.list_index = []
        self.subset = None

        self.components = {
            'menu': {},
            'table': {},
            'graph': {},
            'filter': {},
        }
        self.init_data()
        self.init_components()

        self.skeleton = {
            'navbar': {},
            'body': {}
        }
        self.make_skeleton()

        self.app.layout = html.Div([self.skeleton['navbar'], self.skeleton['body']])
        self.callback_fullscreen_buttons()
        self.callback_generator()

    def init_data(self):
        """
        Method which intialize data from explainer object
        """
        # TODO : add is_loading if necessary
        if hasattr(self.explainer, 'y_pred'):
            self.dataframe = self.explainer.x_pred.copy()
            if isinstance(self.explainer.y_pred, (pd.Series, pd.DataFrame)):
                self.predict_col = self.explainer.y_pred.columns.to_list()[0]
                self.dataframe = self.dataframe.join(self.explainer.y_pred)
            elif isinstance(self.explainer.y_pred, list):
                self.predict_col = ['_predict_']
                self.dataframe = self.dataframe.join(pd.DataFrame(data=self.explainer.y_pred,
                                                                  columns=[self.predict_col],
                                                                  index=self.explainer.x_pred.index))
            else:
                raise TypeError('y_pred must be of type pd.Series, pd.DataFrame or list')

        else:
            raise ValueError(f'y_pred must be set when calling compile function.')

        self.dataframe['_index_'] = self.explainer.x_pred.index
        self.dataframe.rename(columns={f'{self.predict_col}': '_predict_'}, inplace=True)
        col_order = ['_index_', '_predict_'] + \
                    self.dataframe.columns.drop(['_index_', '_predict_']).tolist()

        self.list_index = random.sample(population=self.dataframe.index.tolist(),
                                        k=min(self.max_rows, len(self.dataframe.index.tolist()))
                                        )
        self.dataframe = self.dataframe[col_order].loc[self.list_index].sort_index()

    def init_components(self):
        """
        Initialize components (graph, table, ...) and insert it inside
        components containers which are created by init_skeleton
        """
        # TODO : MAX : Modifier la mise en forme du dropdown (largeur fonction des éléments, en top z-index, éventuellement avec flèche à droite ...)

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
                    width="auto",
                    align="center",
                ),
                dbc.Col(
                    dbc.Collapse(
                        dbc.FormGroup(
                            [
                                dbc.Label("Class to analyse", style={'color': 'white', 'margin': '0px 5px'}),
                                dcc.Dropdown(
                                    id="select_label",
                                    options=[],
                                    value=None,
                                    clearable=False, searchable=False,
                                    style={"verticalAlign": "middle", "zIndex": '1010', "min-width": '200px'}
                                )
                            ],
                            row=True,
                            style={"margin": "0px 0px 0px 5px", "align-items": "center"}
                        ),
                        is_open=True,
                        id='select_collapse'
                    ),
                    width="auto",
                    align="center"
                )
            ],
            no_gutters=True,
            justify="end",
            form=True
        )

        self.adjust_menu()

        self.components['table']['dataset'] = dash_table.DataTable(
            id='dataset',
            data=self.dataframe.to_dict('records'),
            columns=[
                {"name": i,
                 "id": i,
                 } for i in self.dataframe.columns
            ],
            editable=False,
            row_deletable=False,
            style_as_list_view=True,
            fixed_rows={'headers': True, 'data': 0},

            page_size=self.page_size,
            page_action='custom',
            filter_action='custom',
            filter_query='',
            sort_action='custom',
            sort_mode='multi',

            sort_by=[],
            page_current=0,
            active_cell={'row': 0, 'column': 0},
        )
        self.adjust_dataset()

        self.components['graph']['global_feature_importance'] = MyGraph(
            figure=go.Figure(),
            style={'hovermode': 'closest', "max-width": "1000px", "max-height": "1000px", "margin": "auto"},
            id='global_feature_importance'
        )

        self.components['graph']['feature_selector'] = MyGraph(
            figure=go.Figure(),
            style={'height': '22rem', 'clickmode': 'event'},
            id='feature_selector'
        )
        self.components['graph']['feature_selector'].figure['layout'].clickmode = 'event+select'

        self.components['graph']['detail_feature'] = MyGraph(
            figure=go.Figure(),
            style={'height': '12rem'},
            id='detail_feature'
        )

        self.components['filter']['index'] = dbc.FormGroup(
            [
                dbc.Label("Index ", align="center", width=4),
                dbc.Col(
                    dbc.Input(
                        id="index_id", type="text", bs_size="md", placeholder="Id must exist",
                        debounce=True, persistence=True,
                    ),
                ),
            ],
            row=True,
        )

        self.components['filter']['threshold'] = dbc.FormGroup(
            [
                dbc.Label("Threshold", html_for="slider", id='threshold_label'),
                dcc.Slider(
                    min=0, max=self.max_threshold, value=0, step=0.1,
                    marks={f'{mark}': f'{mark}' for mark in range(self.max_threshold + 1)},
                    id="threshold_id",
                )
            ],
            style={'width': '100%'}
        )

        self.components['filter']['max_contrib'] = dbc.FormGroup(
            [
                dbc.Label(
                    "Features to display : ", id='max_contrib_label'),
                dcc.Slider(
                    min=1, max=self.max_features, value=self.max_features, step=1,
                    marks={f'{feat}': f'{feat}' for feat in range(self.max_features + 1)},
                    id="max_contrib_id",
                )
            ],
            style={'width': '100%'}
        )

        self.components['filter']['positive_contrib'] = dbc.FormGroup(
            [
                dbc.Checklist(
                    options=[{"label": "Positive contributions only", "value": 1}],
                    value=[],
                    id="check_id",
                    inline=True,
                )
            ]
        )

        self.components['filter']['masked_contrib'] = dbc.FormGroup(
            [
                dbc.Label(
                    "Feature(s) to mask :"),
                dcc.Dropdown(
                    options=[{'label': key, 'value': value} for key, value in self.explainer.inv_features_dict.items()],
                    value='',
                    multi=True,
                    searchable=True,
                    id="masked_contrib_id"
                ),
            ],
            style={'width': '100%'}
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
                                href="https://github.com/MAIF/Diaphane",
                                target="_blank",
                            ),
                            md=5,
                            align="left"
                        ),
                        dbc.Col(
                            self.components['menu'],
                            md=7,
                            align='right',
                        )
                    ],
                    style={'padding': "5px 15px", "verticalAlign" : "middle"},
                    # no_gutters=True
                )
            ],
            fluid=True,
            style={'height': '50px', 'backgroundColor': self.bkg_color},
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
                                    # style={'height': '27rem'}
                                )
                            ],
                            md=5,
                            align="center",
                            style={'padding': '0px 10px'},
                            # style={'height': '27rem'}
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
                                                    # style={"height": "26rem"}
                                                ),
                                            ],
                                            md=4,
                                            align="center",
                                            # style={'height': '27rem'}
                                        ),
                                    ],
                                ),
                            ],
                            md=7,
                            align="center",
                            style={'padding': '0px 10px'},
                            # style={'height': '27rem'}
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

    def adjust_dataset(self):
        """
        Update dataset layout.
        """
        self.components['table']['dataset'].style_table = {
                          'overflowX': 'scroll',
                          'maxWidth': '10rem',
                          'maxHeight': '22rem',
                          'overflowY': 'scroll',
                          'padding-left': '0rem',  # 1,1
                          'padding-right': '1rem'
                      },
        self.components['table']['dataset'].style_cell = {
                         'minHeight': '50px',
                         'minWidth': '50px', 'width': '50px', 'maxWidth': '180px',
                         'whiteSpace': 'normal'
                     },
        self.components['table']['dataset'].style_data_conditional = [
                                     {
                                         'if': {'column_id': '_index_'},
                                         'backgroundColor': '#d3d3d3',
                                     },
                                     {
                                         'if': {'column_id': '_predict_'},
                                         'backgroundColor': '#d3d3d3',
                                     },
                                 ],
        self.components['table']['dataset'].style_filter_conditional = self.components['table']['dataset'].style_data_conditional,
        self.components['table']['dataset'].style_header_conditional = self.components['table']['dataset'].style_data_conditional

    def adjust_graph(self, graph):
        """
        Override graph from explainer object
        Parameters
        ----------
        graph : Graph component
            Component to be modified
        """
        graph.figure.update_layout(
            autosize=True,
            margin=dict(
                l=50,
                r=10,
                b=0,
                t=50,
                pad=0
            ),
            title={
                'y': 1,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'}
        )
        graph.figure.update_xaxes(title='', automargin=True)
        graph.figure.update_yaxes(title='', automargin=True)

    def draw_component(self, component_type, component_id, title=None):
        """
        Method which return a component from a type and id.
        It's the method to insert component inside component container.
        Parameters
        ----------
        type : string
            Type of the component. Can be table, graph, ...
        id : string
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
                html.I(
                    "fullscreen",
                    className="material-icons tiny",
                    style={'marginTop': '8px', 'marginLeft': '1px'}
                ),
                id=f"ember_{component_id}",
                className="dock-expand",
                **{
                    'data-component-type': component_type,
                }
            )
        )
        return component

    def draw_filter(self):
        """
        Method which return filter components block for local contributions plot.
        Returns
        -------
        list
            list of components
        """
        filter = [
            dbc.Container(
                [
                    dbc.Row([self.components['filter']['index']],
                            align="center",
                            style={"height": "4rem"}
                            ),
                    dbc.Row([self.components['filter']['threshold']],
                            align="center",
                            style={"height": "5rem"}
                            ),
                    dbc.Row([self.components['filter']['max_contrib']],
                            align="center",
                            style={"height": "5rem"}
                            ),
                    dbc.Row([self.components['filter']['positive_contrib']],
                            align="center",
                            style={"height": "4rem"}
                            ),
                    dbc.Row([self.components['filter']['masked_contrib']],
                            align="center",
                            ),
                ],
            ),
        ]
        return filter

    def select_point(self, graph, click_data):
        if click_data:
            curve_id = click_data['points'][0]['curveNumber']
            point_id = click_data['points'][0]['pointIndex']
            for curve in range(len(self.components['graph'][graph].figure['data'])):
                self.components['graph'][graph].figure['data'][curve].selectedpoints = \
                    [point_id] if curve == curve_id else []

    def split_filter_part(self, filter_part):
        operators = [['ge ', '>='],
                     ['le ', '<='],
                     ['lt ', '<'],
                     ['gt ', '>'],
                     ['ne ', '!='],
                     ['eq ', '='],
                     ['contains '],
                     ['datestartswith ']]
        for operator_type in operators:
            for operator in operator_type:
                if operator in filter_part:
                    name_part, value_part = filter_part.split(operator, 1)
                    name = name_part[name_part.find('{') + 1: name_part.rfind('}')]

                    value_part = value_part.strip()
                    v0 = value_part[0]
                    if v0 == value_part[-1] and v0 in ("'", '"', '`'):
                        value = value_part[1: -1].replace('\\' + v0, v0)
                    else:
                        try:
                            value = float(value_part)
                        except ValueError:
                            value = value_part

                    # word operators need spaces after them in the filter string,
                    # but we don't want these later
                    return name, operator_type[0].strip(), value

        return [None] * 3

    @staticmethod
    def check_row(data, index):
        df = pd.DataFrame.from_records(data, index='_index_')
        row = df.index.get_loc(index) if index in list(df.index) else None
        return row

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
                    Output(component_id=f'card_{component_id}', component_property='style'),
                    Output(component_id=f'{component_id}', component_property=component_property),
                ],
                [
                    Input(f'ember_{component_id}', 'n_clicks'),
                    Input(f'ember_{component_id}', 'data-component-type')
                ]
            )
            def ember(click, component_type):
                click = 2 if click is None else click
                toggle_on = True if click % 2 == 0 else False
                if toggle_on:
                    if component_type == 'graph':
                        style_component = {
                            'height': '23rem',
                        }
                    elif component_type == 'table':
                        style_component = {
                            'overflowX': 'scroll',
                            'maxHeight': '22rem',
                        }
                    this_style_card = {
                        'height': '26rem',
                        'width': 'auto',
                        # 'paddingTop': '0.5rem',
                        'zIndex': 900,
                    }

                    return this_style_card, style_component

                else:
                    this_style_card = {
                        'height': 'auto',
                        'width': 'auto',
                        'zIndex': 998,
                        'position': 'fixed',
                        'top': '50px',
                        'bottom': 0,
                        'left': 0,
                        'right': 0,
                        # 'paddingTop': '0.5rem'
                    }
                    style_component = {
                        'height': '87vh',
                        'maxHeight': '87vh',
                    }
                    return this_style_card, style_component

        self.app = app

    def callback_generator(self):
        app = self.app

        @app.callback(
            [
                Output(component_id='dataset', component_property='data'),
                Output(component_id='dataset', component_property='columns')
            ],
            [
                Input('dataset', "page_current"),
                Input('dataset', "page_size"),
                Input('dataset', 'sort_by'),
                Input('dataset', "filter_query")
            ],
            [
                State('index_id', 'value'),
                State('dataset', 'data')
            ]
        )
        def update_datatable(page_current, page_size, sort_by, filter_query, index, data):
            if not filter_query:
                df = self.dataframe
            else:
                filtering_expressions = filter_query.split(' && ')
                df = self.dataframe
                for filter_part in filtering_expressions:
                    col_name, operator, filter_value = self.split_filter_part(filter_part)
                    if operator in ('eq', 'ne', 'lt', 'le', 'gt', 'ge'):
                        # these operators match pandas series operator method names
                        df = df.loc[getattr(df[col_name], operator)(filter_value)]
                    elif operator == 'contains':
                        df = df.loc[df[col_name].str.contains(filter_value)]
                    elif operator == 'datestartswith':
                        # this is a simplification of the front-end filtering logic,
                        # only works with complete fields in standard format
                        df = df.loc[df[col_name].str.startswith(filter_value)]

            if len(sort_by):
                df = df.sort_values(
                    [col['column_id'] for col in sort_by],
                    ascending=[
                        col['direction'] == 'asc'
                        for col in sort_by
                    ],
                    inplace=False
                )

            self.components['table']['dataset'].data = \
                df.iloc[page_current * page_size: (page_current + 1) * page_size].to_dict('records')
            row = self.check_row(data, index)
            if row:
                self.components['table']['dataset'].active_cell = {'row': row, 'column': 1}

            return self.components['table']['dataset'].data, self.components['table']['dataset'].columns

        @app.callback(
            Output(component_id='global_feature_importance', component_property='figure'),
            [
                Input('select_label', 'value'),
                Input('dataset', 'data')
            ],
            [State('global_feature_importance', 'clickData')]
        )
        def update_feature_importance(label, data, clickData):
            ctx = dash.callback_context
            if not ctx.triggered:
                raise PreventUpdate

            if ctx.triggered[0]['prop_id'] == 'select_label.value':
                self.label = label
            else:
                self.list_index = [d['_index_'] for d in data]

            self.components['graph']['global_feature_importance'].figure = \
                self.explainer.plot.features_importance(max_features=self.max_features,
                                                        selection=self.list_index,
                                                        label=self.label
                                                        )
            self.components['graph']['global_feature_importance'].adjust_graph()
            self.components['graph']['global_feature_importance'].figure.layout.clickmode = 'event+select'
            self.select_point('global_feature_importance', clickData)

            return self.components['graph']['global_feature_importance'].figure

        @app.callback(
            Output(component_id='feature_selector', component_property='figure'),
            [
                Input('global_feature_importance', 'clickData'),
                Input('select_label', 'value'),
                Input('dataset', 'data')
            ],
            [State('feature_selector', 'clickData')]
        )
        def update_feature_selector(feature, label, data, clickData):
            ctx = dash.callback_context
            if not ctx.triggered:
                raise PreventUpdate

            if ctx.triggered[0]['prop_id'] == 'select_label.value':
                self.label = label
            elif ctx.triggered[0]['prop_id'] == 'dataset.data':
                if self.subset:
                    self.subset = [d['_index_'] for d in data]
            else:
                self.selected_feature = feature['points'][0]['label']
                self.subset = self.list_index if feature['points'][0]['curveNumber'] == 0 else None

            self.components['graph']['feature_selector'].figure = self.explainer.plot.contribution_plot(
                col=self.selected_feature,
                selection=self.subset,
                label=self.label,
                violin_maxf=self.violin_maxf,
                max_points=self.max_points
            )

            self.select_point('global_feature_importance', clickData)

            if self.subset:
                self.components['graph']['feature_selector'].figure['layout'].title.text += " <b>< Subset ></b>"
            self.components['graph']['feature_selector'].figure['layout'].clickmode = 'event+select'
            self.components['graph']['feature_selector'].adjust_graph()

            return self.components['graph']['feature_selector'].figure

        @app.callback(
            Output(component_id='index_id', component_property='value'),
            [
                Input('feature_selector', 'clickData'),
                Input('dataset', 'active_cell')
            ],
            [
                State('dataset', 'data')
            ]
        )
        # TODO : gestion déselection de la cellule qd clic sur graph
        def update_index_id(click_data, cell, data):
            ctx = dash.callback_context
            if ctx.triggered[0]['prop_id'] == 'feature_selector.clickData':
                selected = click_data['points'][0]['customdata']
                self.components['table']['dataset'].active_cell = None
            elif ctx.triggered[0]['prop_id'] == 'dataset.active_cell':
                if cell:
                    selected = data[cell['row']]['_index_']
                else:
                    raise PreventUpdate
            else:
                raise PreventUpdate

            # self.components['filter']['index'].value = selected
            return selected

        @app.callback(
            dash.dependencies.Output('threshold_label', 'children'),
            [dash.dependencies.Input('threshold_id', 'value')])
        def update_threshold_label(value):
            return f'Threshold : {value}'

        @app.callback(
            dash.dependencies.Output('max_contrib_label', 'children'),
            [dash.dependencies.Input('max_contrib_id', 'value')])
        def update_max_contrib_label(value):
            return f'Features to display : {value}'

        @app.callback(
            Output(component_id='detail_feature', component_property='figure'),
            [
                Input('index_id', 'value'),
                Input('threshold_id', 'value'),
                Input('max_contrib_id', 'value'),
                Input('check_id', 'value'),
                Input('masked_contrib_id', 'value'),
                Input('select_label', 'value')
            ]
        )
        def update_detail_feature(index, threshold, max_contrib, sign, masked, label):
            ctx = dash.callback_context
            if not ctx.triggered:
                raise PreventUpdate

            threshold = threshold if threshold != 0 else None
            sign = (True if sign == [1] else None)
            self.explainer.filter(threshold=threshold,
                                  features_to_hide=masked,
                                  positive=sign,
                                  max_contrib=max_contrib)
            self.components['graph']['detail_feature'].figure = self.explainer.plot.local_plot(index=index,
                                                                                               label=label,
                                                                                               show_masked=True)
            self.components['graph']['detail_feature'].adjust_graph()
            return self.components['graph']['detail_feature'].figure

        @app.callback(
            [
                Output(component_id='dataset', component_property='style_data_conditional'),
                Output(component_id='dataset', component_property='style_filter_conditional'),
                Output(component_id='dataset', component_property='style_header_conditional')
            ],
            [
                Input('index_id', 'value'),
                Input('dataset', 'data'),
            ]
        )
        def select_row(index, data):
            ctx = dash.callback_context
            if not ctx.triggered:
                raise PreventUpdate
            selected = self.check_row(data, index)
            init_style = [
                    {'if': {'column_id': '_index_'}, 'backgroundColor': '#d3d3d3'},
                    {'if': {'column_id': '_predict_'}, 'backgroundColor': '#d3d3d3'}
            ]
            if selected is not None:
                data_style = init_style+[{"if": {"row_index": selected}, "backgroundColor": self.color}]
            else:
                data_style = init_style
            return data_style, init_style, init_style
