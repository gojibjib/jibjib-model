In vggish_train.py we are training a  classifier model for 195 bird classes. We take a pretrained VGGish/ Audioset model by Google and finetune it by letting it iterate during training on more than 80,000 audio samples of 10 second length. 

We obtain the pretrained model by loading the checkpoint of the orignal model provided by Google. The original final layer is cut off and replaced with our own output nodes. 
A directory containing labeled bird songs is iterated over, .wav audiofiles are transformed into spectrogrammes and their corresponding one-hot label vectors and then consumed by the model.
After every epoch a snapshot of the models weights and biases is saved on disk. After, we can restore the model to either do a query or continue with training.

We are deploying the model by enabling Tensorflow Serving to reduce response time drastically. 

The goal of this project is to obtain a machine learning model being able to distinguish several hundred classes of birds by their sound. 




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
