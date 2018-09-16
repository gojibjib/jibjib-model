import tensorflow as tf
import os, sys
from traceback import print_exc

FEATURE_TENSOR = "vggish/input_features:0"
LOGITS = "mymodel/prediction:0"
SAVE_PATH = './save'
MODEL_NAME = 'jibjib_model'
VERSION = 1
SERVE_PATH = './serve/{}/{}'.format(MODEL_NAME, VERSION)

loaded_graph = tf.Graph()

def create_parser():
	import argparse

	arg_desc = "Serializes a saved TensorFlow model into protocol buffer format."

	parser = argparse.ArgumentParser(description=arg_desc)
	parser.add_argument('checkpoint',
						help='The full path to the checkpoint ckpt file and meta file. Example: output/mymodel.ckpt-10 will use the files output/mymodel.ckpt-10 and output/mymodel.ckpt-10.meta',
						type=str)
	parser.add_argument('--features_tensor',
						help='The name of the features Tensor. Default: {}'.format(FEATURE_TENSOR),
						type=str,
						default=FEATURE_TENSOR)
	parser.add_argument('--logits',
						help='The name of the logits Tensor. Default: {}'.format(LOGITS),
						type=str,
						default=LOGITS)
	parser.add_argument('--save_path',
						help='The path to save the serialized model to. Will create on absence. Default: {}'.format(SAVE_PATH),
						type=str,
						default=SAVE_PATH)
	parser.add_argument('--model_name',
						help='The name of the model. Default: {}'.format(MODEL_NAME),
						type=str,
						default=MODEL_NAME)
	parser.add_argument('--model_version',
						help='The model version. Default: {}'.format(VERSION),
						type=str,
						default=VERSION)
	parser.add_argument('--serve_path',
						help='The path where the model will be served from. Default: {}'.format(SERVE_PATH),
						type=str,
						default=SERVE_PATH)
	
	return parser


args = create_parser().parse_args()

# If necessary, create save dir
if not (os.path.exists(args.save_path)):
	try:
		print("Creating {}".format(args.save_path))
		os.makedirs(args.save_path)
	except:
		print("Unable to create {}".format(args.save_path))
		print_exc()
		sys.exit(1)

with tf.Session(graph = loaded_graph) as sess:

	try:
		saver = tf.train.import_meta_graph("{}.meta".format(args.checkpoint))
	except:
		print("Unable to restore meta graph: {}.meta".format(args.checkpoint))
		print_exc()
		sys.exit(1)
	
	try:
		saver.restore(sess, args.checkpoint)
	except:
		print("Unable to restore model {}".format(args.checkpoint))
		print_exc()
		sys.exit(1)

	features_tensor= loaded_graph.get_tensor_by_name(args.features_tensor)
	logits= loaded_graph.get_tensor_by_name(args.logits)


	# create tensors info
	model_input = tf.saved_model.utils.build_tensor_info(features_tensor)
	model_output = tf.saved_model.utils.build_tensor_info(logits)

	# build signature definition
	signature_definition = tf.saved_model.signature_def_utils.build_signature_def(
	inputs={'inputs': model_input},
	outputs={'outputs': model_output},
	method_name= tf.saved_model.signature_constants.PREDICT_METHOD_NAME)


	builder = tf.saved_model.builder.SavedModelBuilder(args.serve_path)

	builder.add_meta_graph_and_variables(
		sess, [tf.saved_model.tag_constants.SERVING],
		signature_def_map={
			tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
				signature_definition
		})

	# Save the model so we can serve it with a model server :)
	builder.save()