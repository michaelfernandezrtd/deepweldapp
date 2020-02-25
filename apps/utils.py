import dash
import dash_cytoscape as cyto
import dash_html_components as html
from dash.dependencies import Input, Output


def pattern_grid(nrow=8, ncol=8):

            edges = []
            size = ncol * nrow
            nodes = [
                {
                    'data': {'id': short, 'label': label},
                    'selectable': selectable, 'grabbable': grabbable,
                    #'classes': "red"
                }
                for short, label, selectable, grabbable in ((str(i), str(i), True, False) for i in range(1, size+1))

            ]

            for i in range(1, size, 1):
                res = i % ncol

                if i < ncol:
                    edges.append({'data': {'source': i, 'target': i + 1}})
                    edges.append({'data': {'source': i, 'target': i + ncol}})
                    edges.append({'data': {'source': i, 'target': i + (ncol + 1)}})

                elif (res != 0 and i > ncol) and i < ncol * (nrow - 1):
                    edges.append({'data': {'source': i, 'target': i + 1}})
                    edges.append({'data': {'source': i, 'target': i - (ncol - 1)}})
                    edges.append({'data': {'source': i, 'target': i + ncol}})
                    edges.append({'data': {'source': i, 'target': i + ncol + 1}})

                elif res == 0:
                    edges.append({'data': {'source': i, 'target': i + ncol}})

                elif i > ncol * (nrow - 1):
                    edges.append({'data': {'source': i, 'target': i + 1}})
                    edges.append({'data': {'source': i, 'target': i - (ncol - 1)}})

            return [nodes, edges]


def stiffener_grid(nrow=8, ncol=8):

    edges = []
    size = ncol * nrow
    nodes = [
        {
            'data': {'id': short, 'label': label},
            'selectable': selectable, 'grabbable': grabbable,
            # 'classes': "red"
        }
        for short, label, selectable, grabbable in ((str(i), str(i), True, False) for i in range(1, size + 1))

    ]

    for i in range(1, size, 1):
        res = i % nrow

        if i <= ncol:
            edges.append({'data': {'source': i, 'target': nrow*ncol - i}})

        elif res == 1:
            edges.append({'data': {'source': i, 'target': i + 1}})

    return [nodes, edges]


my_stylesheet = [
    # Group selectors
    {
        'selector': 'node',
        'style': {
               'content': 'data(label)',
                'shape': 'square',
                'width': 20,
                'height': 20,
                #'background-color': 'blue',
                #'line-color': 'blue'
        }
    },
    {
        "selector": 'edge',
        "style": {
                'curve-style': 'bezier',
                'width': 2,
                'line-color': '#ddd',
                'target-arrow-color': '#ddd'
                }
    },

    {
        "selector": '.highlighted',
        "style": {
                    'background-color': '#61bffc',
                    'line-color': '#61bffc',
                    'target-arrow-color': '#61bffc',
                    'transition-property': 'background-color, line-color, target-arrow-color',
                    'transition-duration': '0.5s'
                 },
    },
    {
        "selector": ".selected",
        "style": {
                    "overlay-color": "black",
                    "overlay-padding": 10,
                    "overlay-opacity": 0.25
                    }
        },
    # Class selectors
    {
        'selector': '.red',
        'style': {
            'background-color': 'red',
            'line-color': 'red'
        }
    },
    {
        'selector': '.blue',
        'style': {
            'background-color': 'blue',
            'line-color': 'blue'
        }
    },
    {
        'selector': '.triangle',
        'style': {
            'shape': 'triangle'
        }
    }
]

styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}


elements = pattern_grid(nrow=8, ncol=8)

app = dash.Dash(__name__)

app.layout = html.Div([
                    cyto.Cytoscape(
                        id='cytoscape-grid',
                        layout={'name': 'grid', 'rows': 3},
                        stylesheet=my_stylesheet,
                        elements=elements,
                        style={
                            'width': '20%', 'height': '10px',
                            'margin-left': "auto", "margin-right": "auto"
                        }
                        ),
                    html.P(id='cytoscape-tapEdgeData-output'),
                    html.P(id='cytoscape-tapNodeData-output')
])


@app.callback(Output('cytoscape-tapNodeData-output', 'children'),
              [Input('cytoscape-grid', 'tapNodeData')])
def displayTapNodeData(data):
    if data:
        return "You recently clicked/tapped the city: " + data['label']


@app.callback(Output('cytoscape-tapEdgeData-output', 'children'),
              [Input('cytoscape-grid', 'tapEdgeData')])
def displayTapEdgeData(data):
    if data:
        return "You recently clicked/tapped the edge between " + data['source'].upper() + " and " + data[
            'target'].upper()


if __name__ == '__main__':
    app.run_server(debug=True)


