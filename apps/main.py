import dash
import dash_cytoscape as cyto
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_table
from apps.utils import pattern_grid, stiffener_grid, my_stylesheet, scatter_plot
import plotly.graph_objs as go
from weldAI.pattern_features_grid import read_distortion, coord_nodes
import numpy as np
import base64
import io
from scipy.interpolate import griddata
from weldAI.model_distortion import model_eval
import pandas as pd
import datetime
from app import app
import json

# if 'DYNO' in os.environ:
#     app_name = os.environ['DASH_APP_NAME']
# else:
app_name = 'deepweld'


distortion_dict = read_distortion(pattern_folder="data/")
filenames = list(distortion_dict.keys())

[ini_coord_x, ini_coord_y, ini_coord_z] = coord_nodes(
    pattern_folder="data/", file_name="Initial-Bottom.rpt")
xi, yi = np.meshgrid(np.unique(ini_coord_x), np.unique(ini_coord_z))

data = {'welding_pattern': np.array([]), 'stiffener_pattern': np.array([])}
nrow = 16
ncol = 16
selected_node = []
selected_node_stiff = []
dff = dict()
[nodes, edges] = pattern_grid(nrow=nrow, ncol=ncol)
[nodes_stiff, edges_stiff] = stiffener_grid(nrow=nrow, ncol=ncol)

layout = html.Div(
    [
            html.Div(html.H1('Deepweld'), style={"text-align": "center"}),
            dcc.Tabs([

                dcc.Tab(label='Project', children=[
                        dcc.Upload(
                                id='upload-data',
                                children=html.Div([
                                    'Drag and Drop or ',
                                    html.A('Open Project File')
                                ]),
                                style={
                                    'width': '100%',
                                    'height': '60px',
                                    'lineHeight': '60px',
                                    'borderWidth': '1px',
                                    'borderStyle': 'dashed',
                                    'borderRadius': '5px',
                                    'textAlign': 'center',
                                    'margin': '10px'
                                },
                                # Allow multiple files to be uploaded
                                multiple=True
                            ),
                            html.Div(id='output-data-upload'),
                ]),

                dcc.Tab(label='Input layout pattern', children=[
                        html.Div([
                            html.Br(),
                            html.Br(),
                            html.Br(),
                            html.H6('Input number of rows and columns:'),
                            dcc.Input(id='row_number',
                                      placeholder='Enter a number of rows...',
                                      type='number',
                                      value='8',
                                      min=4, max=16, step=1,
                                      ),
                            # html.H6('Number of columns'),
                            dcc.Input(id='col_number',
                                      placeholder='Enter a number of cols...',
                                      type='number',
                                      value='8',
                                      min=4, max=16, step=1
                                      ),
                            html.Br(),
                            html.Br(),
                            html.Br(),

                        ], style={
                            'position': 'static',
                            'margin-left': "auto",
                            "margin-right": "auto",
                            "text-align": "center"
                        }),
                        html.Div( children=
                                    [
                                            html.Div(className="six columns", children=
                                                [
                                                              html.Div(html.H4('Build your layout pattern'),
                                                                       style={"text-align": "center"}),
                                                              html.Div(cyto.Cytoscape(
                                                                  id='cytoscape-grid',
                                                                  layout={'name': 'grid', 'rows': nrow,
                                                                          'panningEnabled': False,
                                                                          'zoomingEnabled': False,
                                                                          'userZoomingEnabled': False
                                                                          },
                                                                  stylesheet=my_stylesheet,
                                                                  elements=nodes + edges,
                                                                  style={
                                                                      'width': '400px', 'height': '400px',
                                                                      'margin-left': "auto",
                                                                      'margin-right': "auto",
                                                                      'zoomingEnabled': "False",
                                                                      'userZoomingEnabled': "False",
                                                                      'panningEnabled': "False",
                                                                      'userPanningEnabled': "False",
                                                                      'fit': "True",
                                                                      'border': 'line'
                                                                  }
                                                              )),
                                                              html.P(id='cytoscape-tapNodeData-output', style={
                                                                           'position': 'relative',
                                                                           'margin-left': "auto",
                                                                           "margin-right": "auto",
                                                                           'margin-top': "auto",
                                                                           "text-align": "center"
                                                                       }),
                                                              html.P(id='output-container-button', style={
                                                                           'position': 'relative',
                                                                           'margin-left': "auto",
                                                                           "margin-right": "auto",
                                                                           'margin-top': "auto",
                                                                           "text-align": "center"
                                                                       }),
                                                              html.Br(),
                                                              html.Br(),
                                                              html.Div(html.Button('Clear layout pattern', id='clear_button'),
                                                                       style={
                                                                           'position': 'relative',
                                                                           'margin-left': "auto",
                                                                           "margin-right": "auto",
                                                                           'margin-top': "auto",
                                                                           "text-align": "center"
                                                                       })
                                                ]),
                                            html.Div(className="six columns", children=
                                                [
                                                                html.Div(html.H4('Build your stiffener pattern'),
                                                                       style={"text-align": "center"}),
                                                                    cyto.Cytoscape(
                                                                                  id='cytoscape-stiff',
                                                                                  layout={'name': 'grid', 'rows': nrow,
                                                                                          'panningEnabled': False,
                                                                                          'zoomingEnabled': False,
                                                                                          'userZoomingEnabled': False
                                                                                          },
                                                                                  stylesheet=my_stylesheet,
                                                                                  elements=nodes + edges,
                                                                                  style={
                                                                                      'width': '400px', 'height': '400px',
                                                                                      'margin-left': "auto",
                                                                                      'margin-right': "auto",
                                                                                      'zoomingEnabled': "False",
                                                                                      'userZoomingEnabled': "False",
                                                                                      'panningEnabled': "False",
                                                                                      'userPanningEnabled': "False",
                                                                                      'fit': "True",
                                                                                      'border': 'line'
                                                                          }
                                                                    ),
                                                                    html.P(id='cytoscape-tapNodeData-output-stiff', style={
                                                                        'position': 'relative',
                                                                        'margin-left': "auto",
                                                                        "margin-right": "auto",
                                                                        'margin-top': "auto",
                                                                        "text-align": "center"
                                                                     }),
                                                                    html.P(id='output-container-button-stiff', style={
                                                                        'position': 'relative',
                                                                        'margin-left': "auto",
                                                                        "margin-right": "auto",
                                                                        'margin-top': "auto",
                                                                        "text-align": "center"
                                                                     }),
                                                                    html.Br(),
                                                                    html.Br(),
                                                                    html.Div(html.Button('Clear stiffener pattern', id='clear_button-stiff'),
                                                                               style={
                                                                                   'position': 'relative',
                                                                                   'margin-left': "auto",
                                                                                   "margin-right": "auto",
                                                                                   'margin-top': "auto",
                                                                                   "text-align": "center"
                                                                               })
                                                ]),
                                    ], style={
                                                                 'position': 'relative',
                                                                 'margin-left': "auto",
                                                                 "margin-right": "auto",
                                                                 'margin-top': "auto",
                                                                }

                                                            ),
                                   ]),

                        dcc.Tab(label='Visualization',
                                children=[html.Div([
                                                    html.Div([
                                                        html.Div([
                                                            dcc.Graph(id="distortion-graph",
                                                                     hoverData={"points": [
                                                                                    {
                                                                                      "x": 200,
                                                                                      "y": 200,
                                                                                      "z": -0.15976833810031202,
                                                                                      "curveNumber": 0
                                                                                    }
                                                                                  ]
                                                                                },

                                                            ),
                                                            html.Br(),
                                                            html.Br(),
                                                            html.Div(html.Button('Compute distortion', id='submit_button',),
                                                                        style={
                                                                                     "text-align": "center"
                                                                        },),
                                                        html.Br(),
                                                        html.Br(),

                                                        ], style={'width': 'auto',
                                                                    'display': 'inline-block',
                                                                    'padding': '0 20'
                                                           }),


                                                    ]),



                                                html.Div([
                                                            html.Div(
                                                                        dcc.Graph(id='x-z-scatter',), style={
                                                                                                        'width': '35%',
                                                                                                        'display': 'inline-block',
                                                                                                        'padding': '20 20'
                                                                                                      }
                                                            ),
                                                            html.Br(),
                                                            html.Div(
                                                                        dcc.Graph(id='y-z-scatter'), style={
                                                                                                        'width': '35%',
                                                                                                        'display': 'inline-block',
                                                                                                        'padding': '20 20'
                                                                                                      },
                                                             ),

                                                ], className="row"),

                                ], className='row',  style={
                                                 'position': 'relative',
                                                 'margin-left': "auto",
                                                 "margin-right": "auto",
                                                 'margin-top': 0,
                                                 'margin-bottom': 0
                                                }),

                        ]),
            ]),




    ])


@app.callback(Output('cytoscape-tapNodeData-output', 'children'),
              [Input('cytoscape-grid', 'tapNodeData')])
def displayTapNodeData(data):
    if data:
        node = data['label']
        if node not in selected_node:
            selected_node.append(node)
        else:
            selected_node.remove(node)
        return "Creating pattern " + "--".join([node for node in selected_node])


@app.callback(Output('cytoscape-grid', 'layout'),
              [Input('row_number', 'value')])
def update_layout(layout):
    return {'name': 'grid', 'rows': layout,
                'panningEnabled': False,
                'zoomingEnabled': False,
                'userZoomingEnabled': False
            }


@app.callback(Output('cytoscape-grid', 'elements'),
              [Input('row_number', 'value'), Input('col_number', 'value')],
              [State('cytoscape-grid', 'elements')])
def update_elements(input_nrow, input_ncol, elements):
            if input_nrow is not None and input_ncol is not None:
                node1, edge1 = pattern_grid(int(input_nrow), int(input_ncol))
                return node1 + edge1

            return elements


@app.callback(
    dash.dependencies.Output('output-container-button', 'children'),
    [dash.dependencies.Input('clear_button', 'n_clicks')])
def update_output(n_clicks):
    if n_clicks is not None and len(selected_node) > 0:
        last_node_sequence = selected_node.copy()
        selected_node.clear()

        return "Removed sequence " + "--".join([node for node in last_node_sequence])



@app.callback(Output('cytoscape-tapNodeData-output-stiff', 'children'),
              [Input('cytoscape-stiff', 'tapNodeData')])
def displayTapNodeData(data):
    if data:
        node = data['label']
        if node not in selected_node_stiff:
            selected_node_stiff.append(node)
        else:
            selected_node_stiff.remove(node)
        return "Creating pattern " + "--".join([node for node in selected_node_stiff])


@app.callback(Output('cytoscape-stiff', 'layout'),
              [Input('row_number', 'value')])
def update_layout(layout):
    return {'name': 'grid', 'rows': layout,
                'panningEnabled': False,
                'zoomingEnabled': False,
                'userZoomingEnabled': False
            }


@app.callback(Output('cytoscape-stiff', 'elements'),
              [Input('row_number', 'value'), Input('col_number', 'value')],
              [State('cytoscape-stiff', 'elements')])
def update_elements(input_nrow, input_ncol, elements):
            if input_nrow is not None and input_ncol is not None:
                node2, edge2 = pattern_grid(int(input_nrow), int(input_ncol))
                return node2 + edge2

            return elements


@app.callback(
    dash.dependencies.Output('output-container-button-stiff', 'children'),
    [dash.dependencies.Input('clear_button-stiff', 'n_clicks')])
def update_output(n_clicks):
    if n_clicks is not None and len(selected_node_stiff) > 0:
        print(selected_node_stiff)
        last_node_sequence_stiff = selected_node_stiff.copy()
        selected_node_stiff.clear()

        return "Removed sequence " + "--".join([node for node in last_node_sequence_stiff])


def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return [html.Div([
        html.H5(filename),
        html.H6(datetime.datetime.fromtimestamp(date)),

        dash_table.DataTable(
            data=df.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in df.columns]
        ),

        html.Hr(),  # horizontal line

        # For debugging, display the raw contents provided by the web browser
        #html.Div('Raw Content'),
        #html.Pre(contents[0:200] + '...', style={
        #    'whiteSpace': 'pre-wrap',
        #    'wordBreak': 'break-all'
        #})
    ]), df]


@app.callback(Output('output-data-upload', 'children'),
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename'),
               State('upload-data', 'last_modified')])
def update_output(list_of_contents, list_of_names, list_of_dates):

    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d)[0] for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children



@app.callback(
            Output("distortion-graph", "figure"),
            [Input('submit_button', 'n_clicks'),
             Input('row_number', 'value'),
             Input('col_number', 'value'),
             Input('upload-data', 'contents')
             ],
            [State('upload-data', 'filename'),
             State('upload-data', 'last_modified')]
            )
def update_figure(n_clicks, input_nrow, input_ncol, list_of_contents, list_of_names, list_of_dates):
    input_nrow = int(input_nrow)
    input_ncol = int(input_ncol)

    pattern = np.zeros((1, input_nrow * input_ncol))
    pattern_stiff = np.array([])

    if n_clicks is not None:

        if list_of_names is not None and len(selected_node) == 0:
            data_frames = [
                parse_contents(c, n, d)[1] for c, n, d in
                zip(list_of_contents, list_of_names, list_of_dates)]

            selected_node_file = data_frames[0]['welding_pattern'][~np.isnan(data_frames[0]['welding_pattern'])]
            selected_node_stiff_file = data_frames[0]['stiffener_pattern'][~np.isnan(data_frames[0]['stiffener_pattern'])]

            pattern = np.zeros((1, input_nrow * input_ncol))
            pattern[:, np.array(selected_node_file).astype(int) - 1] = np.array(selected_node_file).astype(int)
            pattern_stiff = np.array([int(i) for i in selected_node_stiff_file]).reshape(len(selected_node_stiff_file))

        else:
            if len(selected_node) > 0:
                pattern[:, np.array(selected_node).astype(int) - 1] = np.array(selected_node).astype(int)

                if len(selected_node_stiff) > 0:
                    pattern_stiff = np.array([int(i) for i in selected_node_stiff]).reshape(len(selected_node_stiff))

        distortion_prediction = model_eval(pattern.astype(int), pattern_stiff.astype(int), input_nrow, input_ncol)

        zi = distortion_prediction.reshape(int(distortion_prediction.shape[1]**0.5),
                                           int(distortion_prediction.shape[1]**0.5))

    else:
        filename = filenames[0]
        zi = griddata((distortion_dict[filename][0], distortion_dict[filename][1]),
                              distortion_dict[filename][2],
                              (xi, yi), method='cubic')
    dff['x'] = xi
    dff['y'] = yi
    dff['z'] = zi

    trace = [go.Surface(y=yi, x=xi, z=zi, colorscale="YlGnBu", opacity=0.8,
                            colorbar={"title": "Distortion (mm)", "len": 0.5, "thickness": 15}, )]

    fig = go.Figure(data=trace,
                    layout=go.Layout(autosize=True, height=400, width=800, margin=dict(r=10, l=10, b=10, t=10),
                                     scene={"xaxis": {'title': "X (mm)",
                                                      "tickfont": {"size": 10}, 'type': "linear", },
                                            "yaxis": {"title": " Y (mm)",
                                                      "tickfont": {"size": 10}, "tickangle": 1},
                                            "zaxis": {
                                                'title': "Z (mm)",
                                                "tickfont": {"size": 10},
                                                "range": [-5, 5]},
                                            "camera": {"eye": {"x": 0.5, "y": 0.5, "z": 0.25}},
                                            "aspectmode": "manual",
                                            "aspectratio": {"x": 1, "y": 1, "z": 0.3}}))

    fig.update_layout(

        title={
            'y': 0.92,
            'x': 0.50,
            # 'xanchor': 'center',
            # 'yanchor': 'bottom',
            "font": {
                "size": 36,
                "color": "black",
                "family": "monospace"
            },
        },
    )

    return fig


@app.callback(
    Output('hover-data', 'children'),
    [Input('distortion-graph', 'hoverData')])
def display_hover_data(hoverData):
    return json.dumps(hoverData, indent=2)

@app.callback(Output('x-z-scatter', 'figure'),
    [Input('distortion-graph', 'hoverData')])
def update_xz_scatter(hoverData):
    XY = dict()
    x = hoverData['points'][0]['x']
    y = hoverData['points'][0]['y']
    XY['X'] = dff['x'][np.where(dff['y'] == y)]
    XY['Y'] = dff['z'][np.where(dff['y'] == y)]
    title = 'XZ scatter'

    return scatter_plot(XY, 'Linear', title)

@app.callback(Output('y-z-scatter', 'figure'),
    [Input('distortion-graph', 'hoverData')])
def update_xz_scatter(hoverData):
    XY = dict()
    x = hoverData['points'][0]['y']
    z = hoverData['points'][0]['z']
    XY['X'] = dff['y'][np.where(dff['x'] == x)]
    XY['Y'] = dff['z'][np.where(dff['x'] == x)]

    title = 'YZ scatter'

    return scatter_plot(XY, 'Linear', title)
