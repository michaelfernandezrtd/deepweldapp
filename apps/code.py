import os

import dash_core_components as dcc
import dash_html_components as html

# if 'DYNO' in os.environ:
#     app_name = os.environ['DASH_APP_NAME']
# else:
app_name = 'deepweld'

layout = html.Div([
    dcc.Markdown(children=open('apps/main.py', 'r').read()),
])
