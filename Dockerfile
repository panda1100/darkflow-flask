# syntax=docker/dockerfile:1.3-labs

FROM tensorflow/tensorflow:1.15.5-gpu-py3-jupyter

RUN <<EOS
apt update
apt install -y git
EOS

RUN <<EOS
pip install \
    Flask \
    Opencv-python-headless \
    Cython \
    Pillow \
    imutils \
    watchdog
EOS

RUN <<EOS
cd /tmp
git clone https://github.com/thtrieu/darkflow.git
cd darkflow
pip install .
cd /tmp
rm -rf darkflow
EOS

ENTRYPOINT ["python"]
CMD ["darkflow-flask.py"]

