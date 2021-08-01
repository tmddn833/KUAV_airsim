from flask import Flask, Response, Request
import time
from flask_cors import CORS
import threading
import logging

app = Flask(__name__)
# app.logger.disabled = True
# log = logging.getLogger('werkzeug')
# log.disabled = True
CORS(app)


@app.route('/')
def index():
    lat_lon = [[79.40124108864444, 21.97974628259094], [80.40124108864444, 20.97974628259094],
               [81.40124108864444, 19.97974628259094]]
    i = int(time.time() * 10) % 3
    return '{"geometry": {"type": "Point", "coordinates": ' + str(
        lat_lon[i]) + '}, "type": "Feature", "properties": {}}'

if __name__ == '__main__':
    thread = threading.Thread(target=app.run, kwargs={'host': '127.0.0.1', 'threaded': True, 'debug': True,
                                                      'use_reloader':False}, daemon=True)
    thread.start()
    print('test')
    while True:
        print("it's working!")
        time.sleep(1)
