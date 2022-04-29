import flask
import time
import os
from flask import request, render_template
from train import Train, Test

server = flask.Flask(__name__, static_folder=os.path.dirname(__file__) + '/static',
                     template_folder=os.path.dirname(__file__) + '/templates')


@server.route('/', methods=['get'])
def index():
    return render_template('index.html', csv_files=os.listdir('static/data/'), model_files=os.listdir('static/model/'))


@server.route('/train', methods=['post'])
def train():
    model = Train(request.form['data'])
    model.train()
    saved_file = model.saveModel()
    return 'Save as %s.' % saved_file


@server.route('/test', methods=['post'])
def test():
    model = Test(request.form['data'])
    return str(model.predict(eval(request.form['inputdata'])))


@server.route('/upload', methods=['post'])
def upload():
    fname = request.files['csv']  # 获取上传的文件
    if fname:
        t = time.strftime('%Y%m%d%H%M%S')
        new_fname = r'static/data/' + t + fname.filename
        fname.save(new_fname)  # 保存文件到指定路径
        return '上传成功！'
    else:
        return '{"msg": "请上传文件！"}'


server.run(port=8000, debug=True)
