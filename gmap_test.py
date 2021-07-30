import pandas as pd
from bokeh.io import output_notebook
import os
api_key = os.environ['GOOGLE_API_KEY']
bokeh_width, bokeh_height = 500,400
lat, lon = 46.2437, 6.0251

from bokeh.io import show
from bokeh.plotting import gmap
from bokeh.models import GMapOptions
import time

gmap_options = GMapOptions(lat=lat, lng=lon,
                           map_type='roadmap', zoom=10)
p = gmap(api_key, gmap_options, title='Pays de Gex',
         width=bokeh_width, height=bokeh_height)
p.circle([lat], [lon], size=10, color='red')
p.circle()
show(p)

