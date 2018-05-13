Vggish_train_demo.py itereates through directories containing .wav-files, gets the labels from the directory names and finetunes a VGGish feature extractor.



Deploy_model.py restores the pretrained TensorFlow model, consumes a .waf file and generates a semantically meaningful, high-level 128-D embeddinga. The embeddings are tfrecord files that can be fed into a downstream classification model later on.
