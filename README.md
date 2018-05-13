Vggish_train_demo.py itereates through directories containing .wav-files, gets the labels from the directory names and finetunes a VGGish feature extractor.

Work in Progress:

Deploying the finetuned model to extraxt  semantically meaningful, high-level 128-D embeddings from arbitrary audio files. The embeddings are tfrecord files that can be fed into a downstream classification model.
