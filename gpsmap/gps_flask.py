from flask import Flask, Response , render_template , jsonify
import time
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

global i
i = 0
@app.route('/gen',methods=['POST'])
def gen():
    global i
    i+=1
    if i % 3== 0:
        i -=3
    print(i)
    lat_lon = [[79.40124108864444, 21.97974628259094],[80.40124108864444, 20.97974628259094],[81.40124108864444, 19.97974628259094]]
    return jsonify({
        'time' : lat_lon[i][0],})



if __name__ == '__main__':
    app.run(host='127.0.0.1', debug=True)