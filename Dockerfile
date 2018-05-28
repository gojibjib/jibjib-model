FROM tensorflow/tensorflow:latest-gpu

WORKDIR /model

COPY requirements-docker.txt ./
COPY *.py ./
COPY *.ckpt ./

RUN pip install -r requirements-docker.txt

VOLUME /model/input
VOLUME /model/output

CMD ["python", "vggish_train.py"]