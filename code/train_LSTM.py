#!/usr/bin/env python3
# train_LSTM.py - Train a Recurrent Convolutional Network to recognize bird voices 

#TODO input_shape only 10 seconds


import tensorflow as tf
import keras 
from keras import Sequential
from keras.models import Sequential, Model
from keras.layers import Input, Dense, TimeDistributed, LSTM, Dropout, Activation, Reshape
from keras.layers import Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Flatten, Conv2D, BatchNormalization, Lambda
from keras.optimizers import Adam, RMSprop
from keras import regularizers
import logging
import keras.backend as K
import os
import sys
import numpy as np 
import vggish_input
import vggish_params
from sklearn.model_selection import train_test_split
import vggish_params
import matplotlib.pyplot as plt

import scipy

import datetime

flags = tf.app.flags
slim = tf.contrib.slim
import time

flags.DEFINE_boolean(
    'debug', False,
    'Sets log level to debug. Default: false (defaults to log level info)')

flags.DEFINE_integer('minibatch_size', 16, 'Number of Mini batches executed per epoch (batch).')

flags.DEFINE_integer('num_classes', 6, 'Number of classes to train on')

flags.DEFINE_integer('sample_length', 10, 'Length of sample')

flags.DEFINE_string(
    'checkpoint', '../input/crnn.h5',
    'Path to the VGGish checkpoint file.')

flags.DEFINE_float('test_size', 0.2, 'Size of test set as chunk of batch')

flags.DEFINE_integer('save_step', 4, 'Defines _after_ how many epochs the model should be saved.')

flags.DEFINE_integer('epochs', 150, 'Defines how many times the entire train set is fed into the model')

#batch_size=16,epochs=100,

flags.DEFINE_string('model_version', "1.0", "Defines the model version. Will be used for pickle output file")

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


def show_summary_stats(history):
    # List all data in history
    print(history.history.keys())

    #First image: Acc during training on X_train and X_test
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    #Second image: Training and test loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()




def load_spectrogram(rootDir):
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
  train_id_list =[]
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
          try:
            signal_example = vggish_input.wavfile_to_examples(path)
          except Exception as e: 
            print(e)
            print("Skipping {}, unable to extract clean signal example".format(fname))
            continue

          # Build own one-hot encoder 
          encoded = np.zeros((FLAGS.num_classes))
          encoded[counter]=1
          encoded=encoded.tolist()

          # Encode each frame of the example, which results in the final label for this file
          #signal_label =np.array([encoded]*signal_example.shape[0])          
          signal_label =np.array(encoded)
          #print("Signal label shape of {}: {}:".format(fname, signal_label.shape))

          # Check if a clean label can be extracted
          if signal_label != []:
            if len(signal_example)  >= FLAGS.sample_length:
              input_labels.append(signal_label)
              input_examples.append(signal_example[:FLAGS.sample_length])
              #print('shape is {}'.format(input_examples.shape))
          else:
            print("Skipping {}, unable extract clean signal label".format(fname))
            continue

    counter +=1
  try:
    with open(output_dir, "wb") as wf:
      pickle.dump(train_id_list, wf)
  except:
    print("Unable to dump into {}".format(output_dir))

  print("Input examples created: {}, Labels created: {}".format(len(input_examples), len(input_labels)))
  return np.array(input_examples), np.array(input_labels)


def LSTM_Model(X):
  

  input_shape = (X[0].shape[0], X[0].shape[1], X[0].shape[2])
  model_input = Input(input_shape, name='input')

  layer =  model_input



  N_LAYERS = 1
  #try 3, then 4 for filter length
  FILTER_LENGTH = 2
  #TODO get values from vggish.params
  CONV_FILTER_COUNT = 56  
  LSTM_COUNT = 96
  NUM_HIDDEN = 64
  L2_regularization = 0.001
  N_DENSE = 3


  layer = Conv2D(
      filters=CONV_FILTER_COUNT,
      kernel_size=FILTER_LENGTH,
      kernel_regularizer=regularizers.l2(L2_regularization),  
      name='conv_{}'.format(i)
      )(layer)

  layer = BatchNormalization(momentum=0.9)(layer)
  layer = Activation('relu')(layer)
  layer = MaxPooling2D(2)(layer)
  layer = Dropout(0.4)(layer)   
  #bring back the reshape, lstm, and dropout
  layer = Reshape(( int(layer.shape[1]), int(layer.shape[2]) * int(layer.shape[3])))(layer)
  layer = LSTM(LSTM_COUNT, return_sequences=False)(layer)
  layer = Dropout(0.4)(layer)



  # N_DENSE Dense Layers
  for i in range(N_DENSE):
    layer = Dense(NUM_HIDDEN, 
      kernel_regularizer=regularizers.l2(L2_regularization), 
      name='dense' + str(i+1))(layer)
    #investing dropout size...
  layer = Dropout(0.1)(layer)

  ## Softmax Output
  layer = Dense(FLAGS.num_classes)(layer)
  layer = Activation('softmax', name='output_realtime')(layer)
  model_output = layer
  model = Model(model_input, model_output)
    
    
  opt = Adam(lr=0.001)
  model.compile(
          loss='categorical_crossentropy',
          optimizer=opt,
           metrics=['accuracy']
      )
    
  print(model.summary())
  return model


if __name__ == '__main__':

  print("Start")
  print("Loading all examples with corresponding labels...")
  all_examples, all_labels = load_spectrogram(os.path.join(data_dir))
  print("Splitting dataset into training test...")
  X_train_entire, X_validation_entire, y_train_entire, y_validation_entire = train_test_split(all_examples, all_labels, test_size=FLAGS.test_size)
  print("Creating Recurrent LSTM model...")
  lstm_model = LSTM_Model(X_train_entire)
  print("Fitting model...")
  history = lstm_model.fit(X_train_entire, y_train_entire, validation_data = (X_validation_entire, y_validation_entire), batch_size=FLAGS.minibatch_size,epochs=FLAGS.epochs,verbose=1,shuffle=True)
  print('Displaying training statistics')
  show_summary_stats(history)