import tensorflow as tf
import vggish_input
import vggish_slim
import vggish_params
from tensorflow.python import pywrap_tensorflow
import numpy as np
from collections import Counter
import time 
from sklearn.preprocessing import normalize

#loading default graph
loaded_graph = tf.Graph()

logits_result = None
my_array = None
with tf.Session(graph=loaded_graph) as sess:

	# restoring saved model
	saver = tf.train.import_meta_graph('./output/model3/jibjib_model.ckpt-49.meta')
	saver.restore(sess, './output/model3/jibjib_model.ckpt-49')
	
	# get necessary tensors by name
	logits = loaded_graph.get_tensor_by_name("mymodel/prediction:0")
	
	_ , array_size = logits.shape

	#extracting the label
	prediction=tf.argmax(logits,1)

	#restore features tensor for feed_dict
	features_tensor= loaded_graph.get_tensor_by_name("vggish/input_features:0")
	
	#load input for query
	print("amsel")
	my_input = vggish_input.wavfile_to_examples("./wav/my_amsel.wav")
	

	start_time = time.time()
	#pred = sess.run([prediction],feed_dict={features_tensor:my_input})
	logits_result= sess.run([logits],feed_dict={features_tensor:my_input})
	print("--- %s seconds ---" % (time.time() - start_time))
	
	#extract probs from logits

	my_array = np.zeros(array_size)
	print(my_array.shape)

	for element in logits_result[0]:
		#accu
		my_array += element*element


my_list = my_array.tolist()
sorted_list = sorted(my_list)

# 1st, 2nd, 3rd accumulated values
first, second, third = sorted_list[-1], sorted_list[-2], sorted_list[-3]

# 1st, 2nd, 3rd Train IDs
first_bird, second_bird, third_bird = my_list.index(first), my_list.index(second), my_list.index(third)

# Getting confidences
sum_acc = sum([first, second, third])
first_conf, second_conf, third_conf = first / sum_acc, second / sum_acc, third / sum_acc

print(first, second, third)
print(first_bird, second_bird, third_bird)
print(first_conf, second_conf, third_conf)