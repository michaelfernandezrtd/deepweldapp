import os
import dash
import dash_cytoscape as cyto
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
from apps.utils import grid, my_stylesheet
import plotly.graph_objs as go
from weldAI.pattern_features_grid import read_distortion, coord_nodes
import numpy as np
from scipy.interpolate import griddata
from weldAI.model_distortion import model_eval


from app import app

# if 'DYNO' in os.environ:
#     app_name = os.environ['DASH_APP_NAME']
# else:
app_name = 'deepweld'

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
distortion_dict = read_distortion(pattern_folder="data/")
filenames = list(distortion_dict.keys())

[ini_coord_x, ini_coord_y, ini_coord_z] = coord_nodes(
    pattern_folder="data/", file_name="Initial-Bottom.rpt")
xi, yi = np.meshgrid(np.unique(ini_coord_x), np.unique(ini_coord_z))

nrow = 16
ncol = 16
selected_node = []
selected_node_stiff = []
[nodes, edges] = grid(nrow=nrow, ncol=ncol)

layout = html.Div(
                    className="row",
                    children=[
                                    html.Div(html.H1('Deepweld'), style={"text-align": "center"}),
                                    html.Div([
                                                html.H6('Input number of rows and columns:'),
                                                dcc.Input(id='row_number',
                                                            placeholder='Enter a number of rows...',
                                                            type='number',
                                                            value='8',
                                                            min=4, max=16, step=1,
                                                            style={
                                                                    'margin-left': "auto",
                                                                    "margin-right": "auto",
                                                                    'align': 'center'
                                                                  }
                                                         ),
                                                #html.H6('Number of columns'),
                                                dcc.Input(id='col_number',
                                                            placeholder='Enter a number of cols...',
                                                            type='number',
                                                            value='8',
                                                            min=4, max=16, step=1,
                                                            style={
                                                                  'margin-left': "auto",
                                                                  "margin-right": "auto",
                                                                  'align': 'center'
                                                          }
                                                         )
                                            ], style={"text-align": "center"}, className="row"),

                                        html.Div(
                                            className="six columns",
                                            children=[
                                                        html.Div([

                                                                html.Div(html.H4('Build your layout pattern'), style={"text-align": "center"}),
                                                                cyto.Cytoscape(
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
                                                                                        'fit': "False",
                                                                                }
                                                                 ),
                                                                html.P(id='cytoscape-tapNodeData-output'),
                                                                html.P(id='output-container-button'),
                                                                html.Button('Clear', id='clear_button',
                                                                         style={
                                                                                "position": "relative",
                                                                                'margin-left': "555px",
                                                                                'margin-right': "auto",
                                                                                "display": "inline-block"}),

                                                        ]),

                                                                html.Div([html.Br(),
                                                                html.Br(),
                                                                html.Div(html.H4('Build your stiffener pattern'), style={"text-align": "center"}),
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
                                                                                                        'fit': "False",
                                                                                                        'border': 'line'
                                                                                                }
                                                                                ),
                                                                                html.P(id='cytoscape-tapNodeData-output-stiff'),
                                                                                html.P(id='output-container-button-stiff'),
                                                                                html.Button('Clear', id='clear_button-stiff', style={
                                                                                                "position": "relative",
                                                                                                'margin-left': "555px",
                                                                                                'margin-right': "auto",
                                                                                                "display": "inline-block"}),


                                                              ], )

                                                    ], style={'position': 'relative',
                                                                              'margin-left': "auto",
                                                                              "margin-right": "auto",
                                                                              'align': 'center',
                                                                                }
                                        ),


                                    html.Div(className="row", children=[

                                                                dcc.Graph(id="distortion-graph",
                                                                          style={
                                                                              'width': '50%', 'height': '800px',
                                                                              'position': 'relative',
                                                                              'margin-left': "auto", "margin-right": "auto"
                                                                          }
                                                                          ),
                                                                html.Button('Compute distortion', id='submit_button')

                                                    ], style={
                                                            'position': 'relative',
                                                            'margin-left': "auto", "margin-right": "auto"
                                                        },
                                            ),

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
                node1, edge1 = grid(int(input_nrow), int(input_ncol))
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
                node2, edge2 = grid(int(input_nrow), int(input_ncol))
                return node2 + edge2

            return elements


@app.callback(
    dash.dependencies.Output('output-container-button-stiff', 'children'),
    [dash.dependencies.Input('clear_button-stiff', 'n_clicks')])
def update_output(n_clicks):
    if n_clicks is not None and len(selected_node_stiff) > 0:
        last_node_sequence_stiff = selected_node_stiff.copy()
        selected_node_stiff.clear()

        return "Removed sequence " + "--".join([node for node in last_node_sequence_stiff])


@app.callback(
            Output("distortion-graph", "figure"),
            [Input('submit_button', 'n_clicks'),
             Input('row_number', 'value'),
             Input('col_number', 'value')])
def update_figure(n_clicks, input_nrow, input_ncol):
    input_nrow = int(input_nrow)
    input_ncol = int(input_ncol)

    if n_clicks is not None:

        if len(selected_node) == 0:
            pattern = np.array([i for i in range(input_nrow*input_ncol)]).reshape(1, input_nrow*input_ncol) + 1

        else:
            pattern = np.zeros((1, input_nrow * input_ncol))
            pattern[:, np.array(selected_node).astype(int) - 1] = np.array(selected_node).astype(int)

        if len(selected_node_stiff) == 0:
            n = int(input_ncol/2)
            k = int(input_nrow/2)
            pattern_stiff = np.hstack(([np.array([i for i in range(n, input_ncol*(input_nrow+1) - (input_ncol-n), input_nrow)]), np.array([i for i in range(k*input_ncol, (k+1)*input_ncol)])]))

        else:
            pattern_stiff = np.array([int(i) for i in selected_node_stiff]).reshape(len(selected_node_stiff))

        distortion_prediction = model_eval(pattern.astype(int), pattern_stiff.astype(int), input_nrow, input_ncol)

        zi = distortion_prediction.reshape(int(distortion_prediction.shape[1]**0.5),
                                           int(distortion_prediction.shape[1]**0.5))

    else:
        filename = filenames[0]
        zi = griddata((distortion_dict[filename][0], distortion_dict[filename][1]),
                              distortion_dict[filename][2],
                              (xi, yi), method='cubic')

    trace = [go.Surface(y=yi, x=xi, z=zi, colorscale="YlGnBu", opacity=0.8,
                            colorbar={"title": "Distortion (mm)", "len": 0.5, "thickness": 15}, )]

    fig = go.Figure(data=trace,
                    layout=go.Layout(autosize=True, height=800, width=1200,
                                     scene={"xaxis": {'title': "X (mm)",
                                                      "tickfont": {"size": 10}, 'type': "linear", },
                                            "yaxis": {"title": " Y (mm)",
                                                      "tickfont": {"size": 10}, "tickangle": 1},
                                            "zaxis": {
                                                'title': "Z (mm)",
                                                "tickfont": {"size": 10},
                                                "range": [-2, 2]},
                                            "camera": {"eye": {"x": 0.5, "y": 0.5, "z": 0.25}},
                                            "aspectmode": "manual",
                                            "aspectratio": {"x": 1, "y": 1, "z": 0.3}}))

    fig.update_layout(
        title={
            'text': "Distortion surface",
            'y': 0.92,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'bottom',
            "font": {
                "size": 36,
                "color": "black",
                "family": "monospace"
            },
        },
    )

    return fig



