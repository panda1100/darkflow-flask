from darkflow.net.build import TFNet
import cv2

import tensorflow as tf
import threading
import numpy as np
import sys
import os
from PIL import Image
import imutils

from flask import Flask, request, redirect, render_template, url_for, flash
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = './static'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

options = {"model": "cfg/yolo.cfg", "load": "bin/yolo.weights", "threshold": 0.1, "gpu": 0.1}
tfnet = TFNet(options)


def allowed_file(filename):
    return '.' in filename and \
            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET','POST'])
def home():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return render_template('darkflow.html', result=predict(filename))
    return render_template('home.html')


@app.route('/<key>')
def sample(key):
    if key in ['computer','dog','eagle','giraffe','horses','office','person','scream']:
        return render_template('darkflow.html', result=predict('sample_' + key + '.jpg'))
    else:
        return render_template('darkflow.html', result=predict('sample_dog.jpg'))


def predict(key):
    imgcv = cv2.imread("./static/{}".format(key))
    pred = tfnet.return_predict(imgcv)
    for p in pred:
        pt1 = (p["bottomright"]["x"], p["bottomright"]["y"])
        pt2 = (p["topleft"]["x"],p["topleft"]["y"])
        cv2.rectangle(imgcv, pt1, pt2, (255,0,0), 3)
    cv2.imwrite('./static/predicted_{}'.format(key), imgcv)
    return key


if __name__ == '__main__':
    app.run(host='0.0.0.0')
