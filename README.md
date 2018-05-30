Vggish_train_demo.py itereates through directories containing .wav-files, gets the labels from the directory names and finetunes a VGGish feature extractor.

Deploy_model.py restores the pretrained TensorFlow model, consumes a .wav file and generates a semantically meaningful, high-level 128-D embedding. The embeddings are `tfrecord` files that can be fed into a downstream classification model later on.

## Training

### Docker

Get the container:

```
# GPU
docker pull obitech/jibjib-model:gpu-latest

# CPU
docker pull obitech/jibjib-model:cpu-latest
```

Get the [audioset](https://github.com/tensorflow/models/tree/master/research/audioset) checkpoint:

```
curl -O https://storage.googleapis.com/audioset/vggish_model.ckpt
```

Copy all training folders / files into `input/data/`

Get the [`bird_id_map.pickle`](github.com/gojibjib/voice-grabber):

```
curl -O ./input/bird_id_map.pickle https://github.com/gojibjib/voice-grabber/raw/master/meta/bird_id_map.pickle
```

Run the container:

```
docker container run --rm -d \
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
pip install -r requirements.txt
```

Get the [audioset](https://github.com/tensorflow/models/tree/master/research/audioset) checkpoint:

```
curl -O https://storage.googleapis.com/audioset/vggish_model.ckpt
```

Copy all training folders / files into `input/data/`

Get the [`bird_id_map.pickle`](github.com/gojibjib/voice-grabber):

```
curl -O ./input/bird_id_map.pickle https://github.com/gojibjib/voice-grabber/raw/master/meta/bird_id_map.pickle
```

Start training:

```
# Use python2
python ./vggish_train.py
```