import tensorflow as tf
import vggish_input
import vggish_params
import numpy as np
import vggish_postprocess
import timeit






def load_model():

	#get input for model: spectogram of audio file
	path = "/Users/Sebastian/Documents/own_projects/animal_voices/models/research/audioset/wav_files/Acrocephalus_schoenobaenus/Acrocephalus_schoenobaenus_2.wav"
	spectrogram = vggish_input.wavfile_to_examples(path)

	with tf.Session() as sess:
		
		#restores model
		new_saver = tf.train.import_meta_graph('my_test_model.ckpt.meta')
		#apparently needs to be in same repo 
		new_saver.restore(sess, tf.train.latest_checkpoint('./'))
		

		#builds graph
		graph = tf.get_default_graph()


		labels = tf.placeholder(tf.float32, shape=(None, _NUM_CLASSES), name='labels')
		
		
		# Locate all the tensors and ops we need for the query.
		features_tensor = sess.graph.get_tensor_by_name(vggish_params.INPUT_TENSOR_NAME)
		#labels_tensor = sess.graph.get_tensor_by_name('mymodel/train/labels:0')
		
		#init special tensor to extract embedding out of model
		embedding_tensor = sess.graph.get_tensor_by_name(vggish_params.OUTPUT_TENSOR_NAME)

		#prepare a postprocessor
		pproc = vggish_postprocess.Postprocessor('vggish_pca_params.npz')

		start = timeit.timeit()

		#query model for features of provided audio input
		[output] = sess.run ([embedding_tensor],feed_dict = {features_tensor: spectrogram})
		postprocessed_batch = pproc.postprocess(output)
		#print(postprocessed_batch)


		# Print the postprocessed embeddings as a SequenceExample, in a similar
		# format as the features released in AudioSet. Each row of the batch of
		# embeddings corresponds to roughly a second of audio (96 10ms frames), and
		# the rows are written as a sequence of bytes-valued features, where each
		# feature value contains the 128 bytes of the whitened quantized embedding.
		seq_example = tf.train.SequenceExample(
			feature_lists=tf.train.FeatureLists(
				feature_list={
					vggish_params.AUDIO_EMBEDDING_FEATURE_NAME:
						tf.train.FeatureList(
							feature=[
								tf.train.Feature(
									bytes_list=tf.train.BytesList(
										value=[embedding.tobytes()]))
									for embedding in postprocessed_batch
							]
						)
					}
			)
		)

		print(seq_example)
		end = timeit.timeit()
		print ("Time needed:")
		print (end - start)


if __name__ == '__main__':
	load_model()