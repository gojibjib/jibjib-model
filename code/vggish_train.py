#!/usr/bin/env python2

# vggish_train.py - Train a to recognize bird voices upon Google's audioset model
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

# Number of epochs
flags.DEFINE_integer(
    'num_batches', 60,
    'Number of batches of examples to feed into the model. Each batch is of '
    'variable size and contains shuffled examples of each class of audio.')

flags.DEFINE_integer('num_mini_batches', 1400, 'Number of Mini batches executed per epoch (batch).')

flags.DEFINE_integer('num_classes', 145, 'Number of classes to train on')

flags.DEFINE_boolean(
    'train_vggish', True,
    'If Frue, allow VGGish parameters to change during training, thus '
    'fine-tuning VGGish. If False, VGGish parameters are fixed, thus using '
    'VGGish as a fixed feature extractor.')

flags.DEFINE_boolean('gpu_enabled', False, 'If enabled, performs different operations on up to 4 GPUs')

flags.DEFINE_boolean('validation', True, 'If enabled, checks against validation set')

flags.DEFINE_string(
    'checkpoint', '../input/vggish_model.ckpt',
    'Path to the VGGish checkpoint file.')

flags.DEFINE_float('test_size', 0.2, 'Size of validation set as chunk of batch')

FLAGS = flags.FLAGS

# Folders
input_dir = os.path.abspath("../input")
data_dir = os.path.join(input_dir, "data/")
output_dir = os.path.abspath("../output")
log_dir = os.path.join(output_dir, "log/")
model_dir = os.path.join(output_dir, "model/")

# Load pickle into bird_id_map, this dict maps Bird_name -> Database ID
# bird_id_map = {}
# bird_id_map_path = os.path.join(input_dir, "bird_id_map.pickle")

# Save train_id_list as pickle, so we can later translate back train IDs/labels (counter) to birds
train_id_list = []
train_id_list_path = os.path.join(output_dir, "train_id_list.pickle")

# print("Loading {}".format(bird_id_map_path))
# pickle.HIGHEST_PROTOCOL
# try:
#   with open(bird_id_map_path, "rb") as rf:
#     bird_id_map = pickle.load(rf)
# except:
#   print("Unable to load {}".format(bird_id_map_path))
#   print_exc()

# We need to check how many classes are present
# _NUM_CLASSES = len([name for name in os.listdir(data_dir) if not os.path.isfile(name) and name != ".empty"])
# FLAGS.num_classes = len([name for name in os.listdir(data_dir) if not os.path.isfile(name) and name != ".empty"])

#load spectrograms into list
def load_spectrogram(rootDir, log):
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

          encoded = np.zeros((FLAGS.num_classes))
          encoded[counter]=1
          #log.info(encoded)
          encoded=encoded.tolist()
          #log.info(encoded)
          signal_label =np.array([encoded]*signal_example.shape[0])
          #log.info(signal_label)          
          if signal_label != []:
            #all good: signal not empty
            input_labels.append(signal_label)
            input_examples.append(signal_example)
          else:
            #signal and corresponding label won't be considered
            log.warn("Skipping {}, unable extract clean signal label".format(fname))
            continue

    counter +=1

  try:
    with open(train_id_list_path, "wb") as wf:
      pickle.dump(train_id_list, wf)
  except:
    log.info("Unable to dump into {}".format(train_id_list_path))

  return input_examples, input_labels

# Shuffling data
def get_random_batches(full_examples,input_labels):
  all_examples = np.concatenate([x for x in full_examples ])
  all_labels = np.concatenate([x for x in input_labels])
  #all_label = scipy.sparse.hstack([x for x in input_labels])
  labeled_examples = list(zip(all_examples,all_labels))
  shuffle(labeled_examples)
  
  # Separate and return the features and labels.
  features = [example for (example, _) in labeled_examples]
  labels = [label for (_, label) in labeled_examples]

  return (features, labels)
  
def main(_):
  # allow_soft_placement gives fallback GPU, log_device_placement=True displays device info
  with tf.Graph().as_default(), tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    now = datetime.datetime.now().isoformat().replace(":", "_")
    fmt = logging.Formatter('%(asctime)s:%(name)s:%(levelname)s:%(message)s',
                            '%Y%m%d-%H%M%S')
    # TF logger
    tflog = logging.getLogger('tensorflow')
    tflog.setLevel(logging.DEBUG)
    tflog_fh = logging.FileHandler(os.path.join(log_dir, "{}-tf.log".format(now)))
    tflog_fh.setLevel(logging.DEBUG)
    tflog_fh.setFormatter(fmt)
    tflog_sh = logging.StreamHandler(sys.stdout)
    tflog_sh.setLevel(logging.DEBUG)
    tflog_sh.setFormatter(fmt)
    tflog.addHandler(tflog_fh)
    tflog.addHandler(tflog_sh)

    # Root logger
    log = logging.getLogger()
    log.setLevel(logging.DEBUG)
    root_fh = logging.FileHandler(os.path.join(log_dir, "{}-run.log".format(now)))
    root_fh.setFormatter(fmt)
    root_fh.setLevel(logging.DEBUG)
    root_sh = logging.StreamHandler(sys.stdout)
    root_sh.setFormatter(fmt)
    root_sh.setLevel(logging.DEBUG)
    log.addHandler(root_fh)
    log.addHandler(root_sh)

    start = time.time()
    log.info("Number of epochs: {}".format(FLAGS.num_batches))
    log.info("Number of classes: {}".format(FLAGS.num_classes))
    log.info("Number of Mini batches: {}".format(FLAGS.num_mini_batches))
    log.info("Validation enabled: {}".format(FLAGS.validation))
    log.info("Size of Validation set: {}".format(FLAGS.test_size))
    log.info("Multi GPU flag set: {}".format(FLAGS.gpu_enabled))

    run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True)

    # Define VGGish.
    embeddings = vggish_slim.define_vggish_slim(FLAGS.train_vggish)

    # Define a shallow classification model and associated training ops on top
    # of VGGish.
    with tf.variable_scope('mymodel'):


      # Add a fully connected layer with 100 units.
      num_units = 100
      fc = slim.fully_connected(embeddings, num_units)
      
      # Add a classifier layer at the end, consisting of parallel logistic
      # classifiers, one per class. This allows for multi-class tasks.
      if FLAGS.gpu_enabled:
        with tf.device("/gpu:0"):
          logits = slim.fully_connected(fc, FLAGS.num_classes, activation_fn=None, scope='logits')
          tf.sigmoid(logits, name='prediction')
      else:
        logits = slim.fully_connected(fc, FLAGS.num_classes, activation_fn=None, scope='logits')
        tf.sigmoid(logits, name='prediction')
      print("viewing logits...")
      print(logits)
      print("#####")

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
        if FLAGS.gpu_enabled:
          with tf.device("/gpu:1"):
            xent = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=logits, labels=labels, name='xent')
            loss = tf.reduce_mean(xent, name='loss_op')
        else:
          xent = tf.nn.sigmoid_cross_entropy_with_logits(
              logits=logits, labels=labels, name='xent')
          loss = tf.reduce_mean(xent, name='loss_op')
          
        tf.summary.scalar('loss', loss)

        # We use the same optimizer and hyperparameters as used to train VGGish.    
        if FLAGS.gpu_enabled: 
          with tf.device("/gpu:2"):  
            optimizer = tf.train.AdamOptimizer(
                learning_rate=vggish_params.LEARNING_RATE,
                epsilon=vggish_params.ADAM_EPSILON)
            optimizer.minimize(loss, global_step=global_step, name='train_op')
        else:
          optimizer = tf.train.AdamOptimizer(
              learning_rate=vggish_params.LEARNING_RATE,
              epsilon=vggish_params.ADAM_EPSILON)
          optimizer.minimize(loss, global_step=global_step, name='train_op')

        #Add evaluation ops
      with tf.variable_scope("evaluation"):
        if FLAGS.gpu_enabled:
          with tf.device("/gpu:3"):
            prediction = tf.argmax(logits,1)
            tf.add_to_collection('prediction', prediction)
            correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(labels,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        else:
          prediction = tf.argmax(logits,1)
          correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(labels,1))
          accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # create a summarizer that summarizes loss and accuracy
    tf.summary.scalar("Accuracy", accuracy)
    #tf.summary.scalar("validation_accuracy", val_accuracy)
    # add average loss summary over entire batch
    tf.summary.scalar("Loss", tf.reduce_mean(xent)) 

    # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
    summary_op = tf.summary.merge_all()

    # TensorBoard stuff
    train_writer = tf.summary.FileWriter(os.path.join(log_dir, "train/"),
                                      sess.graph)
    test_writer = tf.summary.FileWriter(os.path.join(log_dir, "test/"),
                                      sess.graph)
    
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
  
    #loads all input with corresponding label
    #training
    log.info("Loading data set and mapping birds to training IDs...")
    all_examples, all_labels =load_spectrogram(os.path.join(data_dir), log)
    #creates training and test set
    X_train_entire, X_test_entire, y_train_entire, y_test_entire = sk.train_test_split(all_examples, all_labels, test_size=FLAGS.test_size)

    # validation set stays the same 
    (X_test,y_test) = get_random_batches(X_test_entire,y_test_entire)

    # The training loop.
    for step in range(FLAGS.num_batches):
      log.info("######## Epoch {}/{} started ########".format(step + 1, FLAGS.num_batches))      
      # extract random sequences for each example
      # maybe just allow very little variation
      (X_train, y_train) = get_random_batches(X_train_entire,y_train_entire)
      
      # Train on n batches per epoch
      minibatch_n = FLAGS.num_mini_batches
      minibatch_size = len(X_train) / minibatch_n
      #log.info("\nStarting training with {} audio frames\n".format(len(X_train)))
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
        log.info("Loss in minibatch: "+str(loss))
        log.info("Training accuracy in minibatch: "+str(train_acc))

        log.info("(Epoch {}/{}) ==> Minibatch {} finished ...".format(step+1, FLAGS.num_batches, counter))
        print()
        counter += 1

      if FLAGS.validation:
        del summary, loss, num_steps, train_acc, temp, X_train, y_train, minibatch_n
        try:
           del y_train_mini, X_train_mini
        except:
          log.warn("X_train_mini, y_train_mini are already out of scope")

        minibatch_valid_size = 20
        val_acc_entire = 0.
        for j in range(0, len(X_test), minibatch_valid_size):
          X_test_mini = X_test[j:j + minibatch_valid_size]
          y_test_mini = y_test[j:j + minibatch_valid_size]

          summary,_,val_acc,pred,corr_pred = sess.run([summary_op,loss_tensor,accuracy,prediction,correct_prediction], feed_dict={features_tensor: X_test_mini, labels_tensor: y_test_mini},  options=run_options)
          val_acc_entire += val_acc

        

          test_writer.add_summary(summary, step*minibatch_valid_size+j)

        average_val_acc= val_acc_entire/(j/minibatch_valid_size)
        log.info("Validation Accuracy: {}".format(average_val_acc))

      # Save model to disk.
      saver = tf.train.Saver()
      if step % 15 == 0:
        save_path = saver.save(sess, os.path.join(model_dir, "jibjib_model.ckpt"),global_step=step)
        log.info("Model saved to %s" % save_path)

    now = datetime.datetime.now().isoformat().replace(":", "_").split(".")[0]
    end = time.time()
    out = "Training finished after {}s".format(end - start)
    log.info(out)

  
if __name__ == '__main__':
  # Disable stdout buffer
  sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)
  
  tf.app.run()
