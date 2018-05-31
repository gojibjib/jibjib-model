#!/usr/bin/env python2

# vggish_train.py - Train a to recognize bird voices upon Google's audioset model
# https://github.com/tensorflow/models/tree/master/research/audioset for more information.

from __future__ import print_function
from random import shuffle
import math
import numpy as np
import tensorflow as tf
import os 
import numpy as np
from numpy import array
import sklearn.model_selection as sk
import vggish_input
import vggish_params
import vggish_slim
import time 
from numpy import array
import pickle
from traceback import print_exc
import sys
import datetime

flags = tf.app.flags
slim = tf.contrib.slim
import time

# Number of epochs
flags.DEFINE_integer(
    'num_batches', 10,
    'Number of batches of examples to feed into the model. Each batch is of '
    'variable size and contains shuffled examples of each class of audio.')

flags.DEFINE_integer('num_mini_batches', 5, 'Number of Mini batches executed per epoch (batch).')

flags.DEFINE_integer('num_classes', 6, 'Number of classes to train on')

flags.DEFINE_boolean(
    'train_vggish', True,
    'If Frue, allow VGGish parameters to change during training, thus '
    'fine-tuning VGGish. If False, VGGish parameters are fixed, thus using '
    'VGGish as a fixed feature extractor.')

flags.DEFINE_string(
    'checkpoint', 'vggish_model.ckpt',
    'Path to the VGGish checkpoint file.')

FLAGS = flags.FLAGS

# Folders
input_dir = os.path.abspath("./input")
data_dir = os.path.join(input_dir, "data/")
output_dir = os.path.abspath("./output")
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
def load_spectrogram(rootDir):
  counter = 0
  input_examples =[]
  input_labels = []

  for dirName, subdirList, fileList  in os.walk(rootDir):
    bird = os.path.basename(os.path.normpath(dirName))
    if bird == "data":
      continue

    print("{} -> {}".format(bird, counter))
    train_id_list.append(bird)
    for fname in fileList:
      if fname.endswith(".wav"):
          path = os.path.join(dirName, fname)

          #calling vggish function, reads in wav file and returns mel spectrogram
          signal_example = vggish_input.wavfile_to_examples(path)
          
          encoded = np.zeros((FLAGS.num_classes))
          encoded[counter]=1

          encoded=encoded.tolist()
          signal_label =np.array([encoded]*signal_example.shape[0])

          #Shows what classes got what encoding on terminal
          input_examples.append(signal_example)
          input_labels.append(signal_label)
    counter +=1

  try:
    with open(train_id_list_path, "wb") as wf:
      pickle.dump(train_id_list, wf)
  except:
    print("Unable to dump into {}".format(train_id_list_path))

  return input_examples, input_labels

# Shuffling data
def get_random_batches(full_examples,input_labels):
  all_examples = np.concatenate([x for x in full_examples ])
  all_labels = np.concatenate([x for x in input_labels])
  labeled_examples = list(zip(all_examples,all_labels))
  shuffle(labeled_examples)
  
  # Separate and return the features and labels.
  features = [example for (example, _) in labeled_examples]
  labels = [label for (_, label) in labeled_examples]

  return (features, labels)
  
def main(_):
  with tf.Graph().as_default(), tf.Session() as sess:
    start = time.time()

    # Define VGGish.
    embeddings = vggish_slim.define_vggish_slim(FLAGS.train_vggish)

    # Define a shallow classification model and associated training ops on top
    # of VGGish.
    with tf.variable_scope('mymodel'):
      print("Number of epochs: {}".format(FLAGS.num_batches))
      print("Number of classes: {}".format(FLAGS.num_classes))
      print("Number of Mini batches: {}".format(FLAGS.num_mini_batches))

      # Add a fully connected layer with 100 units.
      num_units = 100
      fc = slim.fully_connected(embeddings, num_units)

      # Add a classifier layer at the end, consisting of parallel logistic
      # classifiers, one per class. This allows for multi-class tasks.
      logits = slim.fully_connected(fc, FLAGS.num_classes, activation_fn=None, scope='logits')
      
      tf.sigmoid(logits, name='prediction')

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

        #Add evaluation ops
      with tf.variable_scope("evaluation"):
        prediction = tf.argmax(logits,1)
        correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(labels,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # create a summarizer that summarizes loss and accuracy
    tf.summary.scalar("Accuracy", accuracy)
    #tf.summary.scalar("validation_accuracy", val_accuracy)
    # add average loss summary over entire batch
    tf.summary.scalar("Loss", tf.reduce_mean(xent)) 

    # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
    #summary_op = tf.summary.merge_all()
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
    print("Loading data set and mapping birds to training IDs...")
    all_examples, all_labels =load_spectrogram(os.path.join(input_dir, "data/"))
    #creates training and test set
    X_train_entire, X_test_entire, y_train_entire, y_test_entire = sk.train_test_split(all_examples, all_labels, test_size=0.2)
    
    # The training loop.
    for step in range(FLAGS.num_batches):
      print("######## Epoch {}/{} started ########".format(step + 1, FLAGS.num_batches))      
      # extract random sequences for each example
      # maybe just allow very little variation
      (X_train, y_train) = get_random_batches(X_train_entire,y_train_entire)
      
      #validation set stays the same 
      (X_test,y_test) = get_random_batches(X_test_entire,y_test_entire)
      
      # Train on n batches per epoch
      minibatch_n = FLAGS.num_mini_batches
      minibatch_size = len(X_train) / minibatch_n
      #print("\nStarting training with {} audio frames\n".format(len(X_train)))
      counter = 1
      for i in range(0, len(X_train), minibatch_size):
        print("(Epoch {}/{}) ==> Minibatch {} started ...".format(step+1, FLAGS.num_batches, counter))
        # Get pair of (X, y) of the current minibatch/chunk

        X_train_mini = X_train[i:i + minibatch_size]
        y_train_mini = y_train[i:i + minibatch_size]
        
        [summary,num_steps, loss,_, train_acc,temp] = sess.run([summary_op,global_step_tensor, loss_tensor, train_op,accuracy,prediction],feed_dict={features_tensor: X_train_mini, labels_tensor: y_train_mini})
        train_writer.add_summary(summary, step*minibatch_size+i)
        print("Loss in minibatch: "+str(loss))
        print("Training accuracy in minibatch: "+str(train_acc))
        
        # Check validation accuracy every step
        if i%2 == 0:
          summary,loss,val_acc,pred, corr_pred = sess.run([summary_op,loss_tensor,accuracy,prediction,correct_prediction], feed_dict={features_tensor: X_test, labels_tensor: y_test})
          print("Validation Accuracy: {}".format(val_acc))
          test_writer.add_summary(summary, step*minibatch_size+i)

        print("(Epoch {}/{}) ==> Minibatch {} finished ...".format(step+1, FLAGS.num_batches, counter))
        print()
        counter += 1

    # Save model to disk.
    saver = tf.train.Saver()
    save_path = saver.save(sess, os.path.join(model_dir, "jibjib_model.ckpt"),global_step=2)
    print("Model saved to %s" % save_path)

    now = datetime.datetime.now().isoformat().replace(":", "_").split(".")[0]
    end = time.time()
    out = "Training finished after {}s".format(end - start)
    print(out)

    with open(os.path.join(output_dir, now), "w") as wf:
      wf.write("Training finished after {}\n".format(out))
  
if __name__ == '__main__':
  # Disable stdout buffer
  sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)
  
  tf.app.run()

