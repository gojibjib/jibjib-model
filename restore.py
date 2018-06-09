import tensorflow as tf
import vggish_input
import vggish_slim
import vggish_params
from tensorflow.python import pywrap_tensorflow
import numpy as np


loaded_graph = tf.Graph()


with tf.Session(graph=loaded_graph) as sess:
	# restore save model
	saver = tf.train.import_meta_graph('./output/model/jibjib_model.ckpt-19.meta')
	saver.restore(sess, tf.train.latest_checkpoint('./output/model/.'))
	
	#does random stuff when activated
	#sess.run(tf.global_variables_initializer())

	op_list = sess.graph.get_operations()

	for element in op_list:
		print(element)
	
	
	# get necessary tensors by name
	logits = loaded_graph.get_tensor_by_name("mymodel/prediction:0")
	print(logits)
	prediction=tf.argmax(logits,1)

	
	#features_tensor = sess.graph.get_tensor_by_name(vggish_params.INPUT_TENSOR_NAME)
	features_tensor= loaded_graph.get_tensor_by_name("vggish/input_features:0")
	
	my_input = vggish_input.wavfile_to_examples("/Users/Sebastian/github/jibjib/jibjib-model/input/data/Anas_querquedula/Anas_querquedula_2_2.wav")
	pred = sess.run([prediction],feed_dict={features_tensor:my_input})
	print(pred)
