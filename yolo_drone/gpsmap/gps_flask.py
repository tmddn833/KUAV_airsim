from flask import Flask, Response, Request
import time
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return gen()


def gen():
    lat_lon = [[79.40124108864444, 21.97974628259094], [80.40124108864444, 20.97974628259094],
               [81.40124108864444, 19.97974628259094]]
    i = int(time.time()*10)%3
    return '{"geometry": {"type": "Point", "coordinates": ' + str(lat_lon[i]) + '}, "type": "Feature", "properties": {}}'


if __name__ == '__main__':
<<<<<<< HEAD:gpsmap/gps_flask.py
    app.run(host='127.0.0.1', debug=True)
=======
    thread = threading.Thread(target=app.run, kwargs={'host': '127.0.0.1', 'threaded': True, 'debug': True,
                                                      'use_reloader':False}, daemon=True)
    thread.start()
    # print('test')
    # while True:
    #     print("it's working!")
    #     time.sleep(1)
>>>>>>> 363b7318e9d370f8246055b3f06069d816ed1a40:yolo_drone/gpsmap/gps_flask.py
