import pandas as pd
from bokeh.io import output_notebook
import os
bokeh_width, bokeh_height = 500,400
lat, lon = 46.2437, 6.0251
api_key = 'AIzaSyBJJYlhsZDmnAUGmryqI_RIsw7SQaOF-hE'

from bokeh.io import show
from bokeh.plotting import gmap
from bokeh.models import GMapOptions
import time

def plot(lat, lng, zoom=10, map_type='roadmap'):
    gmap_options = GMapOptions(lat=lat, lng=lng,
                               map_type=map_type, zoom=zoom)
    p = gmap(api_key, gmap_options, title='Pays de Gex',
             width=bokeh_width, height=bokeh_height)
    show(p)
    return p

p = plot(lat, lon)
time.sleep(10)
