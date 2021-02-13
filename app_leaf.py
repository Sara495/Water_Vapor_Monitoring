import dash
import dash_leaflet as dl
import dash_html_components as html
from dash.dependencies import Input, Output

# Cool, dark tiles by Stadia Maps.
url = 'https://tiles.stadiamaps.com/tiles/alidade_smooth_dark/{z}/{x}/{y}{r}.png'
attribution = '&copy; <a href="https://stadiamaps.com/">Stadia Maps</a> '
# Create app.
app = dash.Dash()

app.layout = html.Div([
  dl.Map([dl.TileLayer(url=url, maxZoom=20, attribution=attribution),dl.LayerGroup(id="layer")],
  id="map", style={'width': '100%', 'height': '50vh', 'margin': "auto", "display": "block"}),
  ])




@app.callback(Output("layer", "children"), [Input("map", "click_lat_lng")])
def map_click(click_lat_lng):
    return [dl.Marker(position=click_lat_lng, children=dl.Tooltip("({:.3f}, {:.3f})".format(*click_lat_lng)))]

if __name__ == '__main__':
    app.run_server()   