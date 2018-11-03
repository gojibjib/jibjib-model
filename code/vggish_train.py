#!/usr/bin/env python2

# vggish_train.py - Train a model to recognize bird voices upon Google's audioset model
# https://github.com/tensorflow/models/tree/master/research/audioset for more information.

from __future__ import print_function
from random import shuffle
import sys
import math
import numpy as np
import tensorflow as tf
import os 
import numpy as np
from numpy import array
import sklearn.model_selection as sk
sys.path.insert(0, os.path.abspath("../vggish"))
import logging
import vggish_input
import vggish_params
import vggish_slim
import time 
from numpy import array
import pickle
from traceback import print_exc

import datetime
import scipy

flags = tf.app.flags
slim = tf.contrib.slim
import time

flags.DEFINE_boolean(
    'debug', False,
    'Sets log level to debug. Default: false (defaults to log level info)')

flags.DEFINE_integer(
    'num_batches', 60,
    'Number of batches (epochs) of examples to feed into the model. Each batch is of '
    'variable size and contains shuffled examples of each class of audio.')

flags.DEFINE_integer('num_mini_batches', 1400, 'Number of Mini batches executed per epoch (batch).')

flags.DEFINE_integer('num_classes', 145, 'Number of classes to train on')

flags.DEFINE_boolean(
    'train_vggish', True,
    'If Frue, allow VGGish parameters to change during training, thus '
    'fine-tuning VGGish. If False, VGGish parameters are fixed, thus using '
    'VGGish as a fixed feature extractor.')

flags.DEFINE_boolean('validation', True, 'If enabled, checks against validation set')

flags.DEFINE_string(
    'checkpoint', '../input/vggish_model.ckpt',
    'Path to the VGGish checkpoint file.')

flags.DEFINE_float('test_size', 0.2, 'Size of validation set as chunk of batch')

flags.DEFINE_integer('save_step', 5, 'Defines after how many epochs the model should be saved.')

flags.DEFINE_string('model_version', "1.0", "Defines the model version. Will be used for output files like model ckpt and pickle")

FLAGS = flags.FLAGS

# Folders
input_dir = os.path.abspath("../input")
data_dir = os.path.join(input_dir, "data/")
output_dir = os.path.abspath("../output")
log_dir = os.path.join(output_dir, "log/")
log_dir_test = os.path.join(log_dir, "test/")
log_dir_train = os.path.join(log_dir, "train/")
model_dir = os.path.join(output_dir, "model/")

# Set log level depending on flags
log_level = None
if FLAGS.debug:
  log_level = logging.DEBUG
else:
  log_level = logging.INFO

# Save train_id_list as pickle, so we can later translate back train IDs/labels to birds
train_id_list = []
train_id_list_path = os.path.join(output_dir, "train_id_list-{}.pickle".format(FLAGS.model_version))

def create_dir(path):
  """Checks if a directory exists and creates it, if necessary
  Args:
    path (str): The path of the directory to be checked for existence
  """
  if not os.path.exists(path):
    try:
      print("Creating {}".format(path))
      os.makedirs(path)
    except:
      print("Unable to create {}.".format(path))
      print_exc()

def load_spectrogram(rootDir, log):
  """Iterate over a directory and add each file as an input label

  The file tree should be of the structure <rootDir>/<class>/{data point n}, for example:
  input/data/
  + Accipiter_gentilis
  -   + Accipiter_gentilis__1.wav
  -   + Accipiter_gentilis__2.wav
  + Cygnus_olor
  -   + Cygnus_olor_1.wav
  -   + Cygnus_olor_2.wav
  + Regulus_regulus
      + Regulus_regulus_4_1.wav
      + Regulus_regulus_4_2.wav

  The function iterates over each audio file and extracts both a signal example and a signal label.
  
  A signal example is a 3-D np.array of shape [num_examples, num_frames, num_bands] which represents
  a sequence of examples, each of which contains a patch of log mel
  spectrogram, covering num_frames frames of audio and num_bands mel frequency
  bands, where the frame length is vggish_params.STFT_HOP_LENGTH_SECONDS. The length of num_examples
  corresponds with the duration of the audio file in seconds.

  A signal label is a one-hot encoded vector of the input labels. For example:
  Accipiter_gentilis  --> [1, 0, 0]  
  Cygnus_olor         --> [0, 1, 0] 
  Regulus_regulus     --> [0, 0, 1]
  Each audio file will be split into 0.96s frames, where each frame is one-hot encoded.

  Args:
    rootDir (str): The root directory where dataset is located.
    log: A Python logging object.
  
  Returns:
    (input_examples, input_labels): A tuple of lists, containing feature spectrograms and with corresponding labels.
  """
  counter = 0
  input_examples =[]
  input_labels = []

  for dirName, subdirList, fileList  in os.walk(rootDir):
    bird = os.path.basename(os.path.normpath(dirName))
    if bird == "data":
      continue

    log.info("{} -> {}".format(bird, counter))
    train_id_list.append(bird)
    for fname in fileList:
      if fname.endswith(".wav"):
          path = os.path.join(dirName, fname)

          #calling vggish function, reads in wav file and returns mel spectrogram
          try:
            signal_example = vggish_input.wavfile_to_examples(path)
          except:
            log.warn("Skipping {}, unable to extract clean signal example".format(fname))
            continue

          log.debug("Signal example shape of {}: {}".format(fname, signal_example.shape))

          # Initialize one-hot encoder 
          encoded = np.zeros((FLAGS.num_classes))
          encoded[counter]=1
          encoded=encoded.tolist()

          # Encode each frame of the example, which results in the final label for this file
          signal_label =np.array([encoded]*signal_example.shape[0])          

          log.debug("Signal label shape of {}: {}:".format(fname, signal_label.shape))

          # Check if a clean label can be extracted
          if signal_label != []:
            input_labels.append(signal_label)
            input_examples.append(signal_example)
          else:
            log.warn("Skipping {}, unable extract clean signal label".format(fname))
            continue

    counter +=1

  try:
    with open(train_id_list_path, "wb") as wf:
      pickle.dump(train_id_list, wf)
  except:
    log.warn("Unable to dump into {}".format(train_id_list_path))

  log.debug("Input examples created: {}, Labels created: {}".format(len(input_examples), len(input_labels)))
  return input_examples, input_labels

def get_random_batches(input_examples, input_labels, log):
  """Shuffles up read-in examples and labels.

  The input audio files and the corresponding one-hot encoded labels of their audio frames are first 
  paired up, then shuffled and seperated again. Shuffling is done to prevent a common pattern due to
  reading in audio files in the same order each time and improve the model's ability to generalize.

  Args:
    input_examples (list): A list of 3-D np.arrays of shape [num_example, num_frames, num_bands]
    input_labels (list): A list 2-D np.arrays of shape [encoded_label, num_classes] where each
      example will consist n encoded labels, with n being the number of audio frames the example
      consists of.
    log: A Python logging object

  Returns:
    features (list): A shuffled list of input examples.
    labels (list): A shuffled list of input labels.
  """

  # Create a 3-D np.array of [sum(num_example), num_frames, num_bands]
  all_examples = np.concatenate([x for x in input_examples])

  # Create a 2-D np.array of [sum(encoded_labels), num_classes]
  all_labels = np.concatenate([x for x in input_labels])  
  
  # Pair up examples with corresponding labels in a list, shuffle it
  labeled_examples = list(zip(all_examples,all_labels))
  shuffle(labeled_examples)
  
  # Separate the shuffled list return the features and labels individually
  features = [example for (example, _) in labeled_examples]
  labels = [label for (_, label) in labeled_examples]

  return (features, labels)
  
def main(_):
  # Create folders, if necessary
  for p in (output_dir, log_dir, log_dir_test, log_dir_train, model_dir):
    create_dir(p)

  # allow_soft_placement gives fallback GPU, log_device_placement=True displays device info
  with tf.Graph().as_default(), tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    now = datetime.datetime.now().isoformat().replace(":", "_")
    fmt = logging.Formatter('%(asctime)s:%(name)s:%(levelname)s:%(message)s',
                            '%Y%m%d-%H%M%S')

    # TF logger
    tflog = logging.getLogger('tensorflow')
    tflog.setLevel(log_level)
    tflog_fh = logging.FileHandler(os.path.join(log_dir, "{}-{}-tf.log".format(FLAGS.model_version, now)))
    tflog_fh.setLevel(log_level)
    tflog_fh.setFormatter(fmt)
    tflog_sh = logging.StreamHandler(sys.stdout)
    tflog_sh.setLevel(log_level)
    tflog_sh.setFormatter(fmt)
    tflog.addHandler(tflog_fh)
    tflog.addHandler(tflog_sh)

    # Root logger
    log = logging.getLogger()
    log.setLevel(log_level)
    root_fh = logging.FileHandler(os.path.join(log_dir, "{}-{}-run.log".format(FLAGS.model_version, now)))
    root_fh.setFormatter(fmt)
    root_fh.setLevel(log_level)
    root_sh = logging.StreamHandler(sys.stdout)
    root_sh.setFormatter(fmt)
    root_sh.setLevel(log_level)
    log.addHandler(root_fh)
    log.addHandler(root_sh)

    start = time.time()
    log.info("Model version: {}".format(FLAGS.model_version))
    log.info("Number of epochs: {}".format(FLAGS.num_batches))
    log.info("Number of classes: {}".format(FLAGS.num_classes))
    log.info("Number of Mini batches: {}".format(FLAGS.num_mini_batches))
    log.info("Validation enabled: {}".format(FLAGS.validation))
    log.info("Size of Validation set: {}".format(FLAGS.test_size))
    log.info("Saving model after each {} step".format(FLAGS.save_step))

    run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True)

    # Define VGGish as our convolutional blocks
    embeddings = vggish_slim.define_vggish_slim(FLAGS.train_vggish)

    # Define a shallow classification model and associated training ops on top of VGGish.
    with tf.variable_scope('mymodel'):
      # Add a fully connected layer with 100 units.
      num_units = 100
      fc = slim.fully_connected(embeddings, num_units)
      
      # Add a classifier layer at the end, consisting of parallel logistic
      # classifiers, one per class. This allows for multi-class tasks.
      logits = slim.fully_connected(fc, FLAGS.num_classes, activation_fn=None, scope='logits')
      
      # Use Sigmoid as our activation function
      tf.sigmoid(logits, name='prediction')
      
      log.debug("Logits: {}".format(logits))

      # Add training ops.
      with tf.variable_scope('train'):
        global_step = tf.Variable(
            0, name='global_step', trainable=False,
            collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                         tf.GraphKeys.GLOBAL_STEP])

        # Labels are assumed to be fed as a batch multi-hot vectors, with
        # a 1 in the position of each positive class label, and 0 elsewhere.
        labels = tf.placeholder(
            tf.float32, shape=(None,FLAGS.num_classes), name='labels')
      
        # Cross-entropy label loss.
        xent = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=logits, labels=labels, name='xent')
        loss = tf.reduce_mean(xent, name='loss_op')
          
        tf.summary.scalar('loss', loss)

        # We use the same optimizer and hyperparameters as used to train VGGish.    
        optimizer = tf.train.AdamOptimizer(
            learning_rate=vggish_params.LEARNING_RATE,
            epsilon=vggish_params.ADAM_EPSILON)
        optimizer.minimize(loss, global_step=global_step, name='train_op')

      # Add evaluation ops.
      with tf.variable_scope("evaluation"):
        prediction = tf.argmax(logits,1)
        correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(labels,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Create a summarizer that summarizes loss and accuracy
    # TODO: Fix validation loss summary
    tf.summary.scalar("Accuracy", accuracy)
    # Add average loss summary over entire batch
    tf.summary.scalar("Loss", tf.reduce_mean(xent)) 
    # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
    summary_op = tf.summary.merge_all()

    # TensorBoard stuff
    train_writer = tf.summary.FileWriter(log_dir_train, sess.graph)
    test_writer = tf.summary.FileWriter(log_dir_test, sess.graph)
    
    tf.global_variables_initializer().run()

    # Initialize all variables in the model, and then load the pre-trained
    # VGGish checkpoint.
    sess.run(tf.global_variables_initializer())
    vggish_slim.load_vggish_slim_checkpoint(sess, FLAGS.checkpoint)

    # Locate all the tensors and ops we need for the training loop.
    features_tensor = sess.graph.get_tensor_by_name(vggish_params.INPUT_TENSOR_NAME)
    output_tensor = sess.graph.get_tensor_by_name(vggish_params.OUTPUT_TENSOR_NAME)
    labels_tensor = sess.graph.get_tensor_by_name('mymodel/train/labels:0')
    global_step_tensor = sess.graph.get_tensor_by_name('mymodel/train/global_step:0')
    loss_tensor = sess.graph.get_tensor_by_name('mymodel/train/loss_op:0')
    train_op = sess.graph.get_operation_by_name('mymodel/train/train_op')
  
    # Load all input with corresponding labels
    log.info("Loading data set and mapping birds to training IDs...")
    all_examples, all_labels = load_spectrogram(os.path.join(data_dir), log)
    
    # Create training and test sets
    X_train_entire, X_test_entire, y_train_entire, y_test_entire = sk.train_test_split(all_examples, all_labels, test_size=FLAGS.test_size)

    # Test set stays the same throughout all epochs
    (X_test, y_test) = get_random_batches(X_test_entire, y_test_entire, log)

    # Start training
    for step in range(FLAGS.num_batches):
      log.info("######## Epoch {}/{} started ########".format(step + 1, FLAGS.num_batches))      
      
      # Shuffle the order of input examples to foster generalization
      (X_train, y_train) = get_random_batches(X_train_entire,y_train_entire, log)
      
      # Train on n batches per epoch
      minibatch_n = FLAGS.num_mini_batches
      minibatch_size = len(X_train) / minibatch_n
      if minibatch_size <= 0:
        log.error("Size of minibatch too small ({}), choose smaller number of minibatches or use more classes!".format(minibatch_size))
        sys.exit(1)
    
      counter = 1
      for i in range(0, len(X_train), minibatch_size):
        log.info("(Epoch {}/{}) ==> Minibatch {} started ...".format(step+1, FLAGS.num_batches, counter))
        
        # Get pair of (X, y) of the current minibatch/chunk
        X_train_mini = X_train[i:i + minibatch_size]
        y_train_mini = y_train[i:i + minibatch_size]

        log.info("Size of mini batch (features): {}".format(len(X_train_mini)))
        log.info("Size of mini batch (labels): {}".format(len(y_train_mini)))
        
        [summary,num_steps, loss,_, train_acc,temp] = sess.run([summary_op,global_step_tensor, loss_tensor, train_op,accuracy,prediction],feed_dict={features_tensor: X_train_mini, labels_tensor: y_train_mini}, options=run_options)
        train_writer.add_summary(summary, step*minibatch_size+i)
        log.info("Loss in minibatch: {} ".format(loss))
        log.info("Training accuracy in minibatch: {}".format(train_acc))

        log.info("(Epoch {}/{}) ==> Minibatch {} finished ...\n".format(step+1, FLAGS.num_batches, counter))
        counter += 1

        # Test set mini batching
        minibatch_valid_size = 20
        val_acc_entire = 0.
        for j in range(0, len(X_test), minibatch_valid_size):
          X_test_mini = X_test[j:j + minibatch_valid_size]
          y_test_mini = y_test[j:j + minibatch_valid_size]

          summary,_,val_acc,pred,corr_pred = sess.run([summary_op,loss_tensor,accuracy,prediction,correct_prediction], feed_dict={features_tensor: X_test_mini, labels_tensor: y_test_mini},  options=run_options)
          val_acc_entire += val_acc

          test_writer.add_summary(summary, step*minibatch_valid_size+j)

        average_val_acc= val_acc_entire/(j/minibatch_valid_size)
        log.info("Epoch {} -- Validation Accuracy: {}".format(step+1, average_val_acc))

      # Save model to disk.
      saver = tf.train.Saver()
      if step % FLAGS.save_step == 0:
        save_path = saver.save(sess, os.path.join(model_dir, "jibjib_model-{}.ckpt".format(FLAGS.model_version)),global_step=step)
        log.info("Model saved to {}".format(save_path))

    now = datetime.datetime.now().isoformat().replace(":", "_").split(".")[0]
    end = time.time()
    out = "Training finished after {}s".format(end - start)
    log.info(out)
  
if __name__ == '__main__':
  # Disable stdout buffer
  sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)

  tf.app.run()
