# FROM nvidia/cuda:9.0-runtime
FROM tensorflow/tensorflow

WORKDIR /model

# Install Python
# RUN apt-get install -y python \
#         python-dev \
#         rsync \
#         software-properties-common && \
#         apt-get clean && \
#         rm -rf /var/lib/apt/lists/*

# # Install pip
# RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
#     python get-pip.py && \
#     rm get-pip.py

COPY requirements-docker.txt ./
COPY *.py ./
COPY *.ckpt ./

RUN pip install -r requirements-docker.txt

VOLUME /model/input
VOLUME /model/output

CMD ["python", "vggish_train.py"]
