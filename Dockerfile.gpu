# FROM nvidia/cuda:9.0-runtime
FROM tensorflow/tensorflow:latest-gpu

WORKDIR /model

COPY vggish/ ./vggish
COPY requirements-docker.txt requirements.txt

RUN pip install -r requirements.txt

# Put your code at the end so rebuilds are faster
COPY code/ ./code

VOLUME /model/input
VOLUME /model/output

WORKDIR /model/code
CMD ["python", "vggish_train.py"]
