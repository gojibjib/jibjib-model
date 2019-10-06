# A model for bird sound classification

The model for training the bird classifier.

## Repo layout
The complete list of JibJib repos is:

- [jibjib](https://github.com/gojibjib/jibjib): Our Android app. Records sounds and looks fantastic.
- [deploy](https://github.com/gojibjib/deploy): Instructions to deploy the JibJib stack.
- [jibjib-model](https://github.com/gojibjib/jibjib-model): Code for training the machine learning model for bird classification
- [jibjib-api](https://github.com/gojibjib/jibjib-api): Main API to receive database requests & audio files.
- [jibjib-data](https://github.com/gojibjib/jibjib-data): A MongoDB instance holding information about detectable birds.
- [jibjib-query](https://github.com/gojibjib/jibjib-query): A thin Python Flask API that handles communication with the [TensorFlow Serving](https://www.tensorflow.org/serving/) instance.
- [gopeana](https://github.com/gojibjib/gopeana): A API client for [Europeana](https://europeana.eu), written in Go.
- [voice-grabber](https://github.com/gojibjib/voice-grabber): A collection of scripts to construct the dataset required for model training

## Overview

### CNN for Spectrogram-wise Classification
In vggish_train.py we are training a convolutional classifier model for an arbitrary number of birds. We take a pretrained [VGGish/ Audioset](https://github.com/tensorflow/models/tree/master/research/audioset) model by Google and finetune it by letting it iterate during training on more than 80,000 audio samples of 10 second length. Please read the following papers for more information:

- Hershey, S. et. al., [CNN Architectures for Large-Scale Audio Classification](https://research.google.com/pubs/pub45611.html), ICASSP 2017
- Gemmeke, J. et. al., [AudioSet: An ontology and human-labelled dataset for audio events](https://research.google.com/pubs/pub45857.html), ICASSP 2017

Before you can start, you first need to download a VGGish checkpoint file. You can either use a checkpoint provided by ![Google](https://storage.googleapis.com/audioset/vggish_model.ckpt) or ![our](https://s3-eu-west-1.amazonaws.com/jibjib/model/jibjib_model_raw.tgz) very own model that has been additionally trained for more than 100 hours and 60 epochs on a GPU cluster inside a Docker container.

The original final layer is cut off and replaced with our own output nodes.

During the first training step a directory containing labeled bird songs is iterated over and each .wav file is converted into a spectrogram where the x-axis is the time and the y-axis symbolyzes the frequency. For instance, this is the spectrogram of a golden eagles call:

![mel spectogram](https://github.com/gojibjib/jibjib-model/blob/master/assets/steinadler_50_50.png)

Furthermore, each bird class is one-hot-encoded and then in pairs of features and corresponding labels fed into the model.
After, VGGish's convolutional filters run over each spectrogram and extract meaningful features. The following graphic gives a short overview about how after some convolutions and subpooling the extracted features are then fed into the fully connected layer just like in any other CNN:

![mel spectogram](https://raw.githubusercontent.com/gojibjib/jibjib-model/master/assets/Typical_cnn_spectrogram.png)

After every epoch a snapshot of the models weights and biases is saved on disk. In the next step we can restore the model to either do a query or continue with training.

We are deploying the model by enabling TensorFlow Serving to reduce response time drastically. Check out ![jibjib-query](https://github.com/gojibjib/jibjib-query) to learn more about how we implemented TensorFlow Serving for our model.

### New: Convolutional LSTM for Sequence Classification
In train_LSTM.py we provide a Convolutional LSTM for audio event recognition. Similar to vggish_train.py it performs classification tasks on mel spectrograms. In contrast to vggish_train.py, it does not perform a classification for each spectrogram but analyzes aan array of these matrices and then performs a classification on the entire sequence. C-LSTMs may outperform CNNs when data only contains sparse specific features that don't occure in every timestep.


## Training

### Docker

Get the container:

```
# GPU, needs nvidia-docker installed
docker pull obitech/jibjib-model:latest-gpu

# CPU
docker pull obitech/jibjib-model:latest-cpu
```

Create folders, if necessary:
```
mkdir -p output/logs output/train output/model input/data
```

Get the [audioset](https://github.com/tensorflow/models/tree/master/research/audioset) checkpoint:

```
curl -O input/vggish_model.ckpt https://storage.googleapis.com/audioset/vggish_model.ckpt
```

Copy all training folders / files into `input/data/`


Get the [`bird_id_map.pickle`](github.com/gojibjib/voice-grabber):

```
curl -O input/bird_id_map.pickle https://github.com/gojibjib/voice-grabber/raw/master/meta/bird_id_map.pickle
```

Run the container:

```
docker container run --rm -d \
    --runtime=nvidia \
    -v $(pwd)/input:/model/input \
    -v $(pwd)/output:/model/output \
    obitech/jibjib-model:latest-gpu
```

For quickly starting training run:

```
# GPU
./train_docker.sh

# CPU
./train_docker.sh
```

### Locally

Clone the repo:

```
git clone https://github.com/gojibjib/jibjib-model
```

Install dependencies, **use python2.7**:

```
# CPU training
pip install -r requirements.txt

# GPU training
pip install -r requirements-gpu.txt
```

Copy all training folders / files into `input/data/`

Get the [audioset](https://github.com/tensorflow/models/tree/master/research/audioset) checkpoint:

```
curl -O input/vggish_model.ckpt https://storage.googleapis.com/audioset/vggish_model.ckpt
```

Get the [`bird_id_map.pickle`](github.com/gojibjib/voice-grabber):

```
curl -O input/bird_id_map.pickle https://github.com/gojibjib/voice-grabber/raw/master/meta/bird_id_map.pickle
```

Start training:

```
# Make sure to start the script from the code/ directory !
cd code
python ./vggish_train.py
```

You can then use `modelbuilder.py` to convert the model to protocol buffer.
