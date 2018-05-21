# Copyright 2017 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""A simple demonstration of running VGGish in training mode.

This is intended as a toy example that demonstrates how to use the VGGish model
definition within a larger model that adds more layers on top, and then train
the larger model. If you let VGGish train as well, then this allows you to
fine-tune the VGGish model parameters for your application. If you don't let
VGGish train, then you use VGGish as a feature extractor for the layers above
it.

For this toy task, we are training a classifier to distinguish between three
classes: sine waves, constant signals, and white noise. We generate synthetic
waveforms from each of these classes, convert into shuffled batches of log mel
spectrogram examples with associated labels, and feed the batches into a model
that includes VGGish at the bottom and a couple of additional layers on top. We
also plumb in labels that are associated with the examples, which feed a label
loss used for training.

Usage:
  # Run training for 100 steps using a model checkpoint in the default
  # location (vggish_model.ckpt in the current directory). Allow VGGish
  # to get fine-tuned.
  $ python vggish_train_demo.py --num_batches 100

  # Same as before but run for fewer steps and don't change VGGish parameters
  # and use a checkpoint in a different location
  $ python vggish_train_demo.py --num_batches 50 \
                                --train_vggish=False \
                                --checkpoint /path/to/model/checkpoint
"""

from __future__ import print_function

from random import shuffle
import math
import matplotlib  
matplotlib.use('TkAgg')   
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os 
import numpy as np
import contextlib
import wave
from scipy.io import wavfile
from numpy import array
import sklearn.model_selection as sk
import vggish_input
import vggish_params
import vggish_slim

flags = tf.app.flags
slim = tf.contrib.slim

flags.DEFINE_integer(
    'num_batches', 100,
    'Number of batches of examples to feed into the model. Each batch is of '
    'variable size and contains shuffled examples of each class of audio.')

flags.DEFINE_boolean(
    'train_vggish', True,
    'If Frue, allow VGGish parameters to change during training, thus '
    'fine-tuning VGGish. If False, VGGish parameters are fixed, thus using '
    'VGGish as a fixed feature extractor.')

flags.DEFINE_string(
    'checkpoint', 'vggish_model.ckpt',
    'Path to the VGGish checkpoint file.')

FLAGS = flags.FLAGS

#needs fix
_NUM_CLASSES = 3

def random_frame(spectogram, min_size, max_size):
  x,y,z = spectogram.shape
  if x >=min_size and max_size <=x:
    #picks a random few second frame from spectogram array
    length = np.random.random_integers(low=min_size,high=max_size)
    random_start = np.random.random_integers(low=0,high=x-length)
    return spectogram[random_start:(random_start+length),0:y,0:z]
  else:
    return spectogram

def get_one_hot(target, nb_classes):
  return np.eye(nb_classes)[np.array(target).reshape(-1)]

#load spectrograms into list
def load_spectrogram(rootDir):
  counter = 0
  input_examples =[]
  input_labels = []
   
  for dirName, subdirList, fileList  in os.walk(rootDir):
    print(dirName)
    for fname in fileList:
      if fname.endswith(".wav"):
          path = dirName+"/"+fname
          #print(path)
          #calling vggish function, reads in wav file and returns mel spectrogram
          signal_example = vggish_input.wavfile_to_examples(path)
          encoded = (get_one_hot(counter-1,_NUM_CLASSES))
          signal_label =np.array(encoded)
          #Shows what classes got what encoding on terminal
          print(encoded)
          input_examples.append(signal_example)
          input_labels.append(signal_label)
    counter +=1
  return input_examples, input_labels

def get_random_batches(full_examples,input_labels,start,end):
  
  input_examples=[]
  for element in full_examples:
    #gets just the 5 second sequence for each example 
    signal_example= random_frame(element,start,end)
    input_examples.append(signal_example)

  all_examples = np.concatenate([x for x in input_examples ])
  all_labels = np.concatenate([x for x in input_labels])
  labeled_examples = list(zip(all_examples,all_labels))
  shuffle(labeled_examples)
  # Separate and return the features and labels.
  features = [example for (example, _) in labeled_examples]
  labels = [label for (_, label) in labeled_examples]
  return (features, labels)

  
def main(_):

  with tf.Graph().as_default(), tf.Session() as sess:
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
      logits = slim.fully_connected(
          fc, _NUM_CLASSES, activation_fn=None, scope='logits')
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
            tf.float32, shape=(None, _NUM_CLASSES), name='labels')
      
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

    # add average loss summary over entire batch
    tf.summary.scalar("Loss", tf.reduce_mean(xent)) 

    # merge summaries
    summary_op = tf.summary.merge_all()

    # create saver object
    saver = tf.train.Saver()



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
    print("Load data set...")
    all_examples, all_labels =load_spectrogram("./wav_files/")
    #creates training and test set
    X_train_entire, X_test_entire, y_train_entire, y_test_entire = sk.train_test_split(all_examples, all_labels, test_size=0.2)
    
    # The training loop.
    for step in range(FLAGS.num_batches):      
      #extract random sequences for each example
      #maybe just allow very little variation
      (X_train, y_train) = get_random_batches(X_train_entire,y_train_entire,6,9)
      
      #validation set stays the same 
      (X_test,y_test) = get_random_batches(X_test_entire,y_test_entire,6,9)
      #implementing mini batch
      
      minibatch_size = int(len(X_train)/1)
      
      counter =1
      for i in range(0, len(X_train), minibatch_size):
        # Get pair of (X, y) of the current minibatch/chunk
        X_train_mini = X_train[i:i + minibatch_size]
        y_train_mini = y_train[i:i + minibatch_size]
        print("Step: "+str(step+1)+":")
        print(str(counter)+"/"+str(int(math.ceil(len(X_train)/minibatch_size))))
        
        [num_steps, loss,_, train_acc] = sess.run([global_step_tensor, loss_tensor, train_op,accuracy],feed_dict={features_tensor: X_train_mini, labels_tensor: y_train_mini})
        print("Loss in minibatch: "+str(loss))
        print("Training Acc in minibatch: "+str(train_acc))
        counter +=1

        #every 1 steps validation accuracy
        if (num_steps)%1 == 0:
          val_acc,pred_val = sess.run([accuracy,correct_prediction], feed_dict={features_tensor: X_test, labels_tensor: y_test})
          print("Validation accuracy:", val_acc)
          
          pred = sess.run([prediction],feed_dict={features_tensor:X_test})
          print("###Input###")
          print(y_test)
          print("###Output###")
          print(pred)
          print(pred_val)
          
          """
          fmt = '{:<8}{:<20}{:<20}{}'
          print(fmt.format('', , 'Actual Vector','True/ False'))
          for j, (pred,y_test, pred_val) in enumerate(zip(pred,y_test, pred_val)):
            print(fmt.format(j, pred,y_test, pred_val))
          """
          
          #print( prediction.eval(feed_dict={features_tensor: X_test}))
          #print (sess.run(prediction,{features_tensor:X_test}))
          #print("actual input:")
          #print(y_test)
          #print("Validation prediction:")
          #print(pred_val)

        print("###############################")




    saver = tf.train.Saver()
    #Save the variables to disk.
    saver.save(sess, "./temp/my_test_model.ckpt",global_step=23)
    #print("Model saved in path: %s" % save_path)

if __name__ == '__main__':
  tf.app.run()
  
