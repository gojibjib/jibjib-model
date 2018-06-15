

import tensorflow as tf
from tensorflow.python.framework import graph_util
import os,sys




#loading default graph
loaded_graph = tf.Graph()
graph = tf.get_default_graph()
sess = tf.Session()


saver = tf.train.import_meta_graph('./output/model3/jibjib_model.ckpt-49.meta',clear_devices = True)
input_graph_def = graph.as_graph_def()
saver.restore(sess, './output/model3/jibjib_model.ckpt-49')

#whatever this does...
input_graph_def = graph.as_graph_def()

#might have to change to other name? what do i know...
output_node_names="mymodel/prediction"


output_graph_def = graph_util.convert_variables_to_constants(
	sess, # The session
	input_graph_def, # input_graph_def is useful for retrieving the nodes 
	output_node_names.split(",")  
	)

output_graph="./output/prototype/my_prototype.pb"
with tf.gfile.GFile(output_graph, "wb") as f:
	f.write(output_graph_def.SerializeToString())

sess.close()