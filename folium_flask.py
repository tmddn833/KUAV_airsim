from flask import Flask, Response

import folium
import time
app = Flask(__name__)


@app.route('/')
def index():
    return Response(gen(), mimetype='replace')

def gen():
    i = 0
    while True:
        lat_lon = [[79.40124108864444, 21.97974628259094],[80.40124108864444, 20.97974628259094],[81.40124108864444, 19.97974628259094]]
        i +=1
        i %=3
        time.sleep(0.5)
        yield '{"geometry": {"type": "Point", "coordinates": '+ str(lat_lon[i]) +'}, "type": "Feature", "properties": {}}'



if __name__ == '__main__':
    app.run(host='163.152.127.103', debug=True)