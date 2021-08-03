from flask import Flask
import time
from flask_cors import CORS
import threading
import os
from pathlib import Path
import logging


class GpsFlask():
    def __init__(self, client):
        app = Flask(__name__)
        CORS(app)
        app.logger.disabled = True
        log = logging.getLogger('werkzeug')
        log.disabled = True

        @app.route('/')
        def index():

            return '{"drone":{"type": "Feature", "properties": { "id" :"drone"}, "geometry": {"type": "Point", "coordinates": ' + \
                   str(client.drone_lon_lat) + \
                   '}},"human":{  "type": "Feature", "properties": {"id" :"human"}, "geometry":{"type": "Point", "coordinates": ' \
                   + str(client.human_lon_lat) + '}}}'

        self.thread = threading.Thread(target=app.run, kwargs={'host': '127.0.0.1', 'threaded': True, 'debug': True,
                                                               'use_reloader': False}, daemon=True)
        self.thread.start()
        os.startfile(os.path.dirname(os.path.abspath(__file__)) + "\\templates\\trail.html")
        print('GPS map thread start!')
