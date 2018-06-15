In vggish_train.py we are training a  classifier model for 195 bird classes. We take a pretrained VGGish/ Audioset model by Google and finetune it by letting it iterate during training on more than 80,000 audio samples of 10 second length. 

Before you can start, you first need to download a VGGish checkpoint file. You can either use a checkpoint provided by ![Google](https://storage.googleapis.com/audioset/vggish_model.ckpt) or ![our](https://s3-eu-west-1.amazonaws.com/jibjib/model/jibjib_model_raw.tgz) very own model that has been additionally trained for more than 100 hours and 60 epochs on a GPU cluster.

The original final layer is cut off and replaced with our own output nodes.

During the first training step a directory containing labeled bird songs is iterated over and each .wav file is converted into a spectrogram where the x-axis is the time and the y-axis symbolyzes the frequency. For instance, this is the spectrogram of a golden eagles cry:

![mel spectogram](https://github.com/gojibjib/jibjib-model/blob/master/assets/steinadler_50_50.png)

Furthermore, each bird class is one-hot-encoded and then in pairs of features and corresponding labels fed into the model.
After, VGGishs convolutional filters run over each spectrogram and extract meaningful features. The following graphic gives a short overview about how features get extracted and then fed into the fully connected layer just like in any other CNN:

![mel spectogram](https://raw.githubusercontent.com/gojibjib/jibjib-model/master/assets/Typical_cnn_spectrogram.png)

After every epoch a snapshot of the models weights and biases is saved on disk. In the next step we can restore the model to either do a query or continue with training.

We are deploying the model by enabling Tensorflow Serving to reduce response time drastically. Check out ![the repository](https://github.com/gojibjib/jibjib-query) to learn more about how we implemented Tensorflow Serving for our model. 

## Training

### Docker

Get the container:

```
# GPU
docker pull obitech/jibjib-model:gpu-latest

# CPU, needs nvidia-docker installed
docker pull obitech/jibjib-model:cpu-latest
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
    obitech/jibjib-model:gpu-latest
```

### Locally

Clone the repo:

```
git clone https://github.com/gojibjib/jibjib-model
```

Install dependencies:

```
# Use python2.7
pip install -r requirements.txt
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
