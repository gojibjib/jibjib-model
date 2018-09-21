#!/usr/bin/env python
import tensorflow as tf
import os, sys
from traceback import print_exc

FEATURE_TENSOR = "vggish/input_features:0"
LOGITS = "mymodel/prediction:0"
SAVE_TO = 'serve'
VERSION = '1'
MODEL_NAME = 'jibjib_model'
SAVE_PATH = os.path.abspath(os.path.join(os.getcwd(), SAVE_TO, MODEL_NAME, VERSION))

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
		help='The path to save the serialized model to. Will create on absence. Schema: ./<save_path>/<model_version/<model_name> . Default: {} => {}'.format(SAVE_TO, SAVE_PATH),
		type=str)
	parser.add_argument('--model_version',
		help='The model version. Default: {}'.format(VERSION),
		type=str)
	parser.add_argument('--model_name',
		help='The name of the model. Default: {}'.format(MODEL_NAME),
		type=str)

	return parser

args = create_parser().parse_args()
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

	features_tensor = loaded_graph.get_tensor_by_name(args.features_tensor)
	model_input = tf.saved_model.utils.build_tensor_info(features_tensor)
	
	logits = loaded_graph.get_tensor_by_name(args.logits)
	model_output = tf.saved_model.utils.build_tensor_info(logits)

	# build signature definition
	signature_definition = tf.saved_model.signature_def_utils.build_signature_def(
		inputs={'inputs': model_input},
		outputs={'outputs': model_output},
		method_name= tf.saved_model.signature_constants.PREDICT_METHOD_NAME)

	# Construct save path
	out_path = os.getcwd()
	if args.save_path or args.model_version or args.model_name:
		if args.save_path:
			out_path = os.path.join(out_path, args.save_path)
		else:
			out_path = os.path.join(out_path, SAVE_TO)
		
		if args.model_name:
			out_path = os.path.join(out_path, args.model_name)
		else:
			out_path = os.path.join(out_path, MODEL_NAME)
		
		if args.model_version:
			out_path =  os.path.join(out_path, args.model_version)
		else:
			out_path = os.path.join(out_path, VERSION)
		
	else:
		out_path = SAVE_PATH

	try:
		builder = tf.saved_model.builder.SavedModelBuilder(out_path)
	except:
		print("Unable to create SavedModelBuilder")
		print_exc()
		sys.exit(1)

	builder.add_meta_graph_and_variables(
		sess, [tf.saved_model.tag_constants.SERVING],
		signature_def_map={
			tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
				signature_definition
		})

	builder.save()