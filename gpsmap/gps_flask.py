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
    app.run(host='127.0.0.1', debug=True)
