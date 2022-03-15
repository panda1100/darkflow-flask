from darkflow.net.build import TFNet
import cv2

import tensorflow as tf
import threading
import numpy as np
import sys
from PIL import Image
import imutils

from flask import Flask, request, redirect, render_template, url_for, flash
from werkzeug.utils import secure_filename


app = Flask(__name__)

options = {"model": "cfg/yolo.cfg", "load": "bin/yolo.weights", "threshold": 0.1, "gpu": 0.1}
tfnet = TFNet(options)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/<key>')
def darkflow_predict(key):
    if key in ['computer','dog','eagle','giraffe','horses','office','person','scream']:
        return render_template('darkflow.html', result=sample(key))
    else:
        return render_template('darkflow.html', result=sample('dog'))


def sample(key):
    imgcv = cv2.imread("./static/sample_{}.jpg".format(key))
    pred = tfnet.return_predict(imgcv)
    for p in pred:
        pt1 = (p["bottomright"]["x"], p["bottomright"]["y"])
        pt2 = (p["topleft"]["x"],p["topleft"]["y"])
        cv2.rectangle(imgcv, pt1, pt2, (255,0,0), 3)
    cv2.imwrite('./static/sample_{}_predicted.jpg'.format(key), imgcv)
    result = key
    return result


if __name__ == '__main__':
    app.run(host='0.0.0.0')