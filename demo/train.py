# -*- coding: utf-8 -*
from __future__ import absolute_import
from __future__ import division
#from __future__ import print_function

import os
#os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
#os.environ['TF_CPP_MIN_VLOG_LEVEL']='3'
import sys
import time
import pdb
import random
import math
import threading
import numpy as np
import tensorflow as tf

import config
from reader import Reader


from utils import sparse_tuple_from as sparse_tuple_from
from utils import pad_sequences as pad_sequences
from utils import input_coding as coding
from utils import input_decode as decode


flags = tf.app.flags
logging = tf.logging

# Flags governing the hardware employed for running TensorFlow.
flags.DEFINE_integer('num_gpus', 4, "How many GPUs to use.")

flags.DEFINE_bool("restore_model", True , "Restore model")
#flags.DEFINE_bool("fetch_model", False, "Fetch model")

FLAGS = flags.FLAGS

class CTC_Model():
	"""  Importing and running isolated TF graph """
	def __init__(self, name):
		# Create local graph and use it in the session
		self.graph = tf.Graph()
		self.sess = tf.Session(graph=self.graph, 
			config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
		self.is_training = name == "train"
		self.split_batch_size = int(config.batch_size / FLAGS.num_gpus)
		
		with self.graph.as_default():
			# Build graph
			with tf.device('/cpu:0'):
				with tf.name_scope("helper"):
					self.global_loss = tf.get_variable(
						name="global_loss", shape=[], dtype=tf.float32, initializer=tf.zeros_initializer(), trainable=False)	
					self.global_ler = tf.get_variable(
						name="global_ler", shape=[], dtype=tf.float32, initializer=tf.zeros_initializer(), trainable=False)	

					# Count word num in a batch to measure speed
					self.global_step = tf.Variable(0, name='global_step', trainable=False)  	
				
				with tf.name_scope("input"):
					self.x = tf.placeholder(tf.float32, [None, None, config.num_features])
					# Here we use sparse_placeholder that will generate a SparseTensor required by ctc_loss op.
					self.y = tf.sparse_placeholder(tf.int32)
					# 1d array of size [batch_size]
					self.seq_len = tf.placeholder(tf.int32, [None])

					
					# Split the batch of images and labels for towers.
					x_splits = tf.split(self.x, FLAGS.num_gpus, 0)
					y_splits = tf.sparse_split( sp_input=self.y, num_split=FLAGS.num_gpus , axis = 0 )
					seq_len_split = tf.split(self.seq_len, 4 , 0)	

													
				
				with tf.variable_scope("model"):
					# Create optimizer for training
					self.learning_rate = tf.placeholder(dtype=tf.float32, shape=[], name='lr')
					
					with tf.variable_scope("tower"):
						# Calculate the gradients for each model tower.
						tower_grads = []
						tower_losses = []
						tower_lers = []
						decoded_paths = []
						decoded_inputs = []
						decoded_targets = []
						for i in xrange(FLAGS.num_gpus):
							with tf.device('/gpu:%d' % i):
								# Force all Variables to reside on the CPU.
								#with slim.arg_scope([slim.model_variable, slim.variable], device='/cpu:0'):
								loss,ler,decoded_path = self._tower_loss( x_splits[i], y_splits[i] , seq_len_split[i] )
								#loss = tf.Print(loss,[loss],message = "this is loss",summarize=128)
								tower_losses.append(loss)
								tower_lers.append(ler)

								# Reuse variables for the next tower.
								tf.get_variable_scope().reuse_variables()
								
								if self.is_training:	
									# grads = optimizer.compute_gradients(loss)
									# Keep track of the gradients across all towers.
									# tower_grads.append(grads)
									continue
								else:
									#if len(decoded_paths) != i:
									#	time.sleep(0.01)
									#else:
									decoded_paths.append(decoded_path)
									decoded_inputs.append(x_splits[i])
									decoded_targets.append(tf.sparse_tensor_to_dense(y_splits[i]))
				
					temp_total_loss = tf.add_n(tower_losses)
					self.total_loss = tf.div(temp_total_loss,config.batch_size)
					self.total_ler = tf.add_n(tower_lers)
					self.total_decoded_paths = decoded_paths
					self.total_decoded_inputs = decoded_inputs
					self.total_decoded_targets = decoded_targets 
					
					if self.is_training:
						# We must calculate the mean of each gradient. Note that this is the
						# synchronization point across all towers.
						#grads = self._average_gradients(tower_grads)
						#_grads, _vars = zip(*grads)
						#_grads, _ = tf.clip_by_global_norm(_grads, config.max_grad_norm)
						# Apply the gradients to adjust the shared variables.
						#self.train_op = optimizer.apply_gradients(zip(_grads, _vars), global_step=self.global_step)
						with tf.device('/gpu:3'):
							self.train_op = tf.train.MomentumOptimizer(self.learning_rate,0.9).minimize( self.total_loss , global_step=self.global_step )
						#optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
						#optimizer = tf.train.AdamOptimizer()
					else:
						# We must accumulate the loss and word num for validation phase at every step
						self.global_loss_update = tf.assign_add(self.global_loss, self.total_loss )
						self.global_ler_update = tf.assign_add(self.global_ler, self.total_ler )

				model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="model")
				# Add global_step to restore if necessary
				model_vars.append(self.global_step)
				self.model_saver = tf.train.Saver(model_vars)
				self.init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
		
			
	#def _get_state_variables(self, batch_size, cell):
	#	# For each layer, get the initial state and make a variable out of it
	#	# to enable updating its value.
	#	state_variables = []
	#	for state_c, state_h in cell.zero_state(batch_size, tf.float32):
	#		state_variables.append(tf.contrib.rnn.LSTMStateTuple(
	#			tf.Variable(state_c, trainable=False),
	#			tf.Variable(state_h, trainable=False)))
	#	# Return as a tuple, so that it can be fed to dynamic_rnn as an initial state
	#	return tuple(state_variables)

	#def _get_state_update_op(self, state_variables, new_states):

	#	# Add an operation to update the train states with the last state tensors
	#	update_ops = []
	#	for state_variable, new_state in zip(state_variables, new_states):
	#		# Assign the new state to the state variables on this layer
	#		update_ops.extend(
	#			[state_variable[0].assign(new_state[0]),
	#			state_variable[1].assign(new_state[1])])
	#	# Return a tuple in order to combine all update_ops into a single operation.
	#	# The tuple's actual value should not be used.
	#	return tf.tuple(update_ops)	

	def _tower_loss( self , inputs, targets ,seq_len ):
		"""
		Calculate the total loss on a single tower running the LSTM model.
		"""
		#with tf.name_scope("input"):
		#	with tf.device("/cpu:0"):
		#		split_batch_size, padding_len , _ = inputs.get_shape().as_list()
		#		#y_flat = tf.reshape(targets, [-1])

		with tf.name_scope("lstm"):
				def lstm_cell(layer_size):
					# Defining the cell
					# Can be:
					#   tf.nn.rnn_cell.RNNCell
					#   tf.nn.rnn_cell.GRUCell
					lstm_cell = tf.contrib.rnn.LSTMCell(layer_size, state_is_tuple=True)
					cell = tf.contrib.rnn.DropoutWrapper(
						lstm_cell, 
						input_keep_prob=config.input_keep_prob,
						output_keep_prob=config.output_keep_prob,
						state_keep_prob=config.lstm_keep_prob) if self.is_training else lstm_cell
					return lstm_cell				

				## Stacking rnn cells
				#stack = tf.contrib.rnn.MultiRNNCell([cell] * num_layers,state_is_tuple=True)
				#stack = tf.contrib.rnn.MultiRNNCell([lstm_cell(config.layer_size[i]) for i in range(config.num_layer)],state_is_tuple=True)
				stack = tf.contrib.rnn.MultiRNNCell([lstm_cell(config.layer_size)],state_is_tuple=True)

				# For each layer, get the initial state. states will be a tuple of LSTMStateTuples.
				#state = self._get_state_variables(self.split_batch_size, stack )

				## The second output is the last state and we will no use that
				#outputs, last_state = tf.nn.dynamic_rnn(stack, inputs, seq_len, dtype=tf.float32,initial_state=state)
				outputs, last_state = tf.nn.dynamic_rnn(stack, inputs, seq_len, dtype=tf.float32 )

				# Add an operation to update the train states with the last state tensors.
				#update_state = self._get_state_update_op(state, last_state)

				#outputs = tf.concat(outputstuple, 2)
				shape = tf.shape(inputs)
				batch_s, max_timesteps = shape[0], shape[1]

				# Reshaping to apply the same weights over the timesteps
				outputs = tf.reshape(outputs, [-1, config.num_hidden])


				# Truncated normal with mean 0 and stdev=0.1
				# Tip: Try another initialization
				# see https://www.tensorflow.org/versions/r0.9/api_docs/python/contrib.layers.html#initializers
				W = tf.get_variable("W",[config.num_hidden,config.num_classes],initializer=tf.contrib.layers.xavier_initializer())

				#tf.summary.histogram("/weights",W) #可视化观看变量

				# Zero initialization
				# Tip: Is tf.zeros_initializer the same?
				b = tf.get_variable("b",[config.num_classes],initializer=tf.zeros_initializer())
				#tf.summary.histogram("/biases",b) #可视化观看变量

				# Doing the affine projection
				logits = tf.matmul(outputs,W)+b

				# Reshaping back to the original shape
				logits = tf.reshape(logits, [batch_s, -1, config.num_classes])

				# Time major
				logits = tf.transpose(logits, (1, 0, 2))

				with tf.name_scope("loss"):
						#with tf.control_dependencies(update_state):
						loss = tf.nn.ctc_loss(targets, logits, seq_len,preprocess_collapse_repeated=True,ignore_longer_outputs_than_inputs=True)
						#tf.summary.histogram("/loss",loss) #可视化观看变量
				cost = tf.reduce_sum(loss)

				# Option 2: tf.nn.ctc_beam_search_decoded (it's slower but you'll get better results)
				decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, seq_len)
				decoded_paths = []
				if not self.is_training:
						print "logits",logits
						decoded_beam, log_prob_beam = tf.nn.ctc_beam_search_decoder(logits, seq_len, top_paths=10)
						decoded_path = decoded_beam
						for item in decoded_path:
								decoded_paths.append(tf.sparse_tensor_to_dense(item))

				# Inaccuracy: label error rate
				with tf.name_scope("ler"):
						ler = tf.reduce_sum(tf.edit_distance(tf.cast(decoded[0], tf.int32),targets))
						#tf.summary.histogram("/ler",ler) #可视化观看变量
				return cost,ler,decoded_paths


	#def _average_gradients(self, tower_grads):
	#	"""Calculate the average gradient for each shared variable across all towers.
	#	"""
	#	average_grads = []
	#	for grad_and_vars in zip(*tower_grads):
	#		# Note that each grad_and_vars looks like the following:
	#		# ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
	#		grads = []
	#		for g, _ in grad_and_vars:
	#			# Add 0 dimension to the gradients to represent the tower.
	#			expanded_g = tf.expand_dims(g, 0)
	#		
	#			# Append on a 'tower' dimension which we will average over below.
	#			grads.append(expanded_g)
	#		
	#		# Average over the 'tower' dimension.
	#		grad = tf.concat(grads, 0)
	#		grad = tf.reduce_mean(grad, 0)
	#		
	#		# Keep in mind that the Variables are redundant because they are shared
	#		# across towers. So .. we will just return the first tower's pointer to
	#		# the Variable.
	#		v = grad_and_vars[0][1]
	#		grad_and_var = (grad, v)
	#		average_grads.append(grad_and_var)
	#	return average_grads
		
	def init(self):	
		self.sess.run(self.init_op)		
		print("Variables initialized ...")
		
	def restore(self, model_dir):
		# Should call init before restore model in case some variables not in this ckpt
		ckpt_path = tf.train.latest_checkpoint(model_dir)
		if ckpt_path:
			self.model_saver.restore(self.sess, tf.train.latest_checkpoint(model_dir))
			print("Read model parameters from %s" % tf.train.latest_checkpoint(model_dir))
		else:
			print("model doesn't exists")

	def save(self, model_path):
		self.model_saver.save(self.sess, model_path)

	def run(self, data, epoch_num, is_pingce , learning_rate=None):
		data_x, data_y = data
		# Padding input to max_time_step of this batch
		batch_train_inputs, batch_train_seq_len = pad_sequences(data_x)
		# Converting to sparse representation so as to to feed SparseTensor input
		batch_train_targets = sparse_tuple_from(data_y)

		#if epoch_num%config.epcho_num_for_test == 0:
		#get pingce result
		if is_pingce and epoch_num % 5  == 0:
			self.get_pingce_result(batch_train_inputs , batch_train_targets , batch_train_seq_len, learning_rate , epoch_num )

		if self.is_training:
			#start = time.time()
			return  self.sess.run([self.total_loss, self.total_ler ,self.global_step, self.train_op], 
				feed_dict={ self.x:  batch_train_inputs , self.y: batch_train_targets , self.learning_rate: learning_rate, self.seq_len: batch_train_seq_len})

		else:
			return self.sess.run([self.global_loss_update,self.global_ler_update], 
								feed_dict={ self.x: batch_train_inputs, 
														self.y: batch_train_targets, 
														self.learning_rate: learning_rate, 
														self.seq_len: batch_train_seq_len})

	def get_pingce_result(self, batch_train_inputs , batch_train_targets , batch_train_seq_len , learning_rate ,epoch_num ):
		decoded_path,decode_input,decode_target = self.sess.run([self.total_decoded_paths,
																														self.total_decoded_inputs,
																														self.total_decoded_targets], 
																									feed_dict={ self.x:  batch_train_inputs ,
																															 self.y: batch_train_targets, 
																															 self.learning_rate: learning_rate, 
																															 self.seq_len: batch_train_seq_len})
		loss = self.sess.run( self.total_decoded_inputs , feed_dict={ self.x:  batch_train_inputs ,
                                                               self.y: batch_train_targets, 
                                                               self.learning_rate: learning_rate, 
                                                               self.seq_len: batch_train_seq_len} )
		dictnory = self.get_dictnory(config.dictdir)
		output_count=0
		for GPU_count in range(len(decoded_path)):
			if output_count > 100:
					break 
			GPU_beams = decoded_path[GPU_count]
			GPU_input = decode_input[GPU_count]
			GPU_target = decode_target[GPU_count]
			for i, seq in enumerate(GPU_beams[0]):
				print "Sequence\t",output_count
				print "\tEpcho",epoch_num,"Origin:\t",GPU_target[i],"\t",decode(GPU_input[i]),"\t",
				for j in range(len(GPU_target[i])):
						if GPU_target[i][j] < 1 or GPU_target[i][j] >= (config.num_classes-1):
								continue
						else:
								print("%s\t" % dictnory[GPU_target[i][j]]),
				print ""
				output_count += 1
				print  "\tEpcho",epoch_num,'Decoded:',
				for topn in range(len(GPU_beams)):
					seq = GPU_beams[topn][i]
					print "\t\t\tTop",topn,"\t",
					for item in seq:
							if item == -1 :
									print("%s" % "?"),
									continue
							elif item == 0:
									#print("%s" % "!"),
									continue
							print dictnory[item],"\t",
					print ""
				print ""
	 

	def get_global_loss(self):
		return self.sess.run(self.global_loss)
	def get_global_ler(self):
		return self.sess.run(self.global_ler)

	#def get_dictnory(self,dictdir):
	#	dictnory = {}  
	#	file = open(dictdir) #读字典，用于解码结果可视化
	#	dictcount = 0 
	#	while 1:
	#			#print dictcount
	#			dictcount += 1
	#			line = file.readline()
	#			if not line:
	#					break
	#			linelist = line.strip().split();
	#			if len(linelist) != 4:
	#					continue
	#			dictnory[int(linelist[2])] = linelist[0]
	#	return dictnory

	def get_dictnory(self,dictdir):
		dictnory = {}  
		file = open(dictdir) #读字典，用于解码结果可视化
		dictcount = 0 
		while 1:
				#print dictcount
				dictcount += 1
				line = file.readline()
				if not line:
						break
				linelist = line.strip().split();
				dictnory[dictcount] = linelist[0]
		return dictnory


def main(_):
	
	assert config.batch_size % FLAGS.num_gpus == 0, ('Batch size must be divisible by number of GPUs')
	split_batch_size = int(config.batch_size / FLAGS.num_gpus)
	#Collect data file information
	data_dir = os.path.join(config.data_path, config.dataset)
	train_filename = "ctc.train.txt"
	train_file = os.path.join(config.data_path, "train", train_filename)
	valid_filename = "ctc.valid.txt"
	valid_file = os.path.join(config.data_path, "valid", valid_filename)
	test_filename = "ctc.test.txt"
	test_file = os.path.join(config.data_path, "test", test_filename)
	
	reader = Reader(data_dir, config.vocab_size)	#The vocab_size is useless in this experiment;	
	
	train_model = CTC_Model("train")
	valid_model = CTC_Model("valid")

	# Init train and valid model first
	train_model.init()
	learning_rate = 1e-2
	epoch_num = 0
	valid_loss_history = []
	
	if FLAGS.restore_model:
		#train_model.init()
		train_model.restore(config.model_dir)
		#test_data_gen = reader.get_batch_from_file_ctc(test_file, config.batch_size)
		#while True:
		#	try:
		#		next_test_batch = test_data_gen.next()
		#		train_model.run( next_test_batch , 0 , True)
		#	except StopIteration:
		#		train_model.run( next_test_batch , 0 , True)
		#		break
		#print "test  done!"
		#return

		#start_time = time.time()
		#valid_model.init()
		#valid_model.restore(config.model_dir)
		##valid_data_gen = reaCTeget_batch_from_file(valid_file, config.batch_size, config.num_step)
		#valid_data_gen = reader.get_batch_from_file_ctc(test_file, config.batch_size)
		#valid_step = 0 
		#while True:
		#	try:    
		#		next_valid_batch = valid_data_gen.next()
		#		valid_model.run( next_valid_batch ,epoch_num , True )
		#		valid_step+=1
		#		print "valid_step",valid_step

		#	except StopIteration:
		#		valid_model.run( next_valid_batch , epoch_num , True)
		#		end_time = time.time()
		#		print("valid is finished in {} seconds".format(end_time-start_time))
		#		break

		#return


	#learning_rate = 1.0	
	#train_data_gen = reader.get_batch_from_file(train_file, config.batch_size, config.num_step)
	train_data_gen = reader.get_batch_from_file_ctc(train_file, config.batch_size)

	while True:
		epoch_num += 1
		epoch_start_time = time.time()
		# Training phase
		while True:
			try:					
				start_time = time.time()
				batch_loss, batch_ler ,train_step, _ = train_model.run( train_data_gen.next(),
																																epoch_num, 
																																False , 
																																learning_rate=learning_rate ) 
				end_time = time.time()
				#print "batch time=",end_time-start_time;
			
			except StopIteration:
				print("One epoch data is finished")
				# Refresh generator for next epoch
				train_data_gen = reader.get_batch_from_file_ctc(train_file, config.batch_size)
				print("Saving model...")
				# Save graph and model parameters
				model_path = os.path.join(config.model_dir, config.model_name+"_"+str(train_step))
				train_model.save(model_path)	
				count = 0
				break
		
			if train_step % config.step_per_log == 0:
				#batch_word_num = config.batch_size * config.num_step
				#train_speed = batch_word_num // (end_time-start_time)
				#train_ppl = np.exp(batch_loss / batch_word_num)
				time_gap = (end_time - start_time)*config.step_per_log
				print("Train step: {} Train loss: {} Train ler: {:.2f} Train time: {}.sec".format(
					train_step, batch_loss ,batch_ler , time_gap))
			
		# Validation phase
		print("start validation")
		start_time = time.time()
		# Restore model parameter from training
		valid_model.init()
		valid_model.restore(config.model_dir)
		#valid_data_gen = reaCTeget_batch_from_file(valid_file, config.batch_size, config.num_step)
		valid_data_gen = reader.get_batch_from_file_ctc(valid_file, config.batch_size)
		valid_step = 0
		while True:
			try:					
				next_valid_batch = valid_data_gen.next()
				valid_model.run( next_valid_batch ,epoch_num , False)
				valid_step+=1

			except StopIteration:
				valid_model.run( next_valid_batch , epoch_num , True)
				end_time = time.time()
				print("valid is finished in {} seconds".format(end_time-start_time))
				break
	
		#valid_ppl = np.exp(valid_model.get_global_loss()/(valid_step*config.batch_size*config.num_step))
		valid_loss = valid_model.get_global_loss()
		valid_ler = valid_model.get_global_ler()
		epoch_end_time = time.time()
		time_gap = epoch_end_time - epoch_start_time
		print("Epoch: {} Valid loss: {:.2f} Valid ler: {:.2f} Valid time:{}.sec".format(epoch_num,
		 																																								valid_loss/(config.batch_size), 
																																										valid_ler/(config.batch_size)
																																										,time_gap))
		#valid_ppl_history.append(valid_ppl)
		valid_loss_history.append(valid_loss)
			
		# If converged, finish training
		if len(valid_loss_history) >= 3:
			if valid_loss_history[-1] > valid_loss_history[-2]:
				learning_rate = learning_rate * config.decay_rate
			if valid_loss_history[-1] > max(valid_loss_history[-3:]):
				print("Training is converged")
				break

				
if __name__ == "__main__":
	tf.app.run()

