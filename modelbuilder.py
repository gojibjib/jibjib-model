import tensorflow as tf
import os

SAVE_PATH = './save'
MODEL_NAME = 'jibjib_model'
VERSION = 1
SERVE_PATH = './serve/{}/{}'.format(MODEL_NAME, VERSION)

#checkpoint = tf.train.latest_checkpoint(SAVE_PATH)

loaded_graph = tf.Graph()


with tf.Session(graph = loaded_graph) as sess:

	saver = tf.train.import_meta_graph('./output/model3/jibjib_model.ckpt-49.meta')
	saver.restore(sess, './output/model3/jibjib_model.ckpt-49')


	features_tensor= loaded_graph.get_tensor_by_name("vggish/input_features:0")
	logits= loaded_graph.get_tensor_by_name('mymodel/prediction:0')


	# create tensors info
	model_input = tf.saved_model.utils.build_tensor_info(features_tensor)
	model_output = tf.saved_model.utils.build_tensor_info(logits)



	# build signature definition
	signature_definition = tf.saved_model.signature_def_utils.build_signature_def(
	inputs={'inputs': model_input},
	outputs={'outputs': model_output},
	method_name= tf.saved_model.signature_constants.PREDICT_METHOD_NAME)


	builder = tf.saved_model.builder.SavedModelBuilder(SERVE_PATH)

	builder.add_meta_graph_and_variables(
		sess, [tf.saved_model.tag_constants.SERVING],
		signature_def_map={
			tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
				signature_definition
		})
	# Save the model so we can serve it with a model server :)
	builder.save()