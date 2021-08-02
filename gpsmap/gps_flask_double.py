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
    lat_lon = [127.026508, 37.581933]
    d = [0.01, 0, -0.01]
    i = int(time.time() * 10) % 3
    j = (i + 1) % 3
    return '{"drone":{"type": "Feature", "properties": { "id" :"drone"}, "geometry": {"type": "Point", "coordinates": ' +\
           str([lat_lon[0]+d[i],lat_lon[1]+d[i]]) + \
           '}},"human":{  "type": "Feature", "properties": {"id" :"human"}, "geometry":{"type": "Point", "coordinates": ' \
           + str([lat_lon[0]-d[j],lat_lon[1]+d[j]]) + '}}}'

# TODO 저기 properties에 id가 아닌 다른걸로 바꾸면 실행이 안된다..? + 라인에 색을 입히는 방법 + 마커를 아이콘으로 바꾸는법


if __name__ == '__main__':
    thread = threading.Thread(target=app.run, kwargs={'host': '127.0.0.1', 'threaded': True, 'debug': True,
                                                      'use_reloader': False}, daemon=True)
    thread.start()
    print('test')
    while True:
        print("it's working!")
        time.sleep(1)
