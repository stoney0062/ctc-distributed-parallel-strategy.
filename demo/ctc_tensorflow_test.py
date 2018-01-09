#coding:utf-8
#  Compatibility imports
from __future__ import absolute_import
from __future__ import division
#from __future__ import print_function

import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_VISIBLE_DEVICES']='0' 

import math
import time
import sys
import os 
import random

import tensorflow as tf
import scipy.io.wavfile as wav
import numpy as np

from six.moves import xrange as range

#os.environ["TF_CPP_MIN_LOG_LEVEL"]='3' #only show Error

#try:
#    from python_speech_features import mfcc
#except ImportError:
#    print("Failed to import python_speech_features.\n Try pip install python_speech_features.")
#    raise ImportError

from utils import sparse_tuple_from as sparse_tuple_from
from utils import pad_sequences as pad_sequences
from utils import input_coding as coding
from utils import input_decode as decode

def fake_data(num_examples, num_features, num_labels, min_size = 10, max_size=100):

    # Generating different timesteps for each fake data
    timesteps = np.random.randint(min_size, max_size, (num_examples,))

    # Generating random input
    inputs = np.asarray([np.random.randn(t, num_features).astype(np.float32) for t in timesteps])

    # Generating random label, the size must be less or equal than timestep in order to achieve the end of the lattice in max timestep
    labels = np.asarray([np.random.randint(0, num_labels, np.random.randint(1, inputs[i].shape[0], (1,))).astype(np.int64) for i, _ in enumerate(timesteps)])

    return inputs, labels

# Constants
SPACE_TOKEN = '<space>'
SPACE_INDEX = 0
FIRST_INDEX = ord('a') - 1  # 0 is reserved to space

# Some configs
num_features = 27
# Accounting the 0th indice +  space + blank label = 28 characters
#num_classes = ord('z') - ord('a') + 1 + 1 + 1

# Hyper-parametnum_epochs = 100000
num_hidden = 1024
num_layers = 1
batch_size = 64
initial_learning_rate = 1e-2
momentum = 0.9
count = 0 #输入数据行数
num_classes = 5002 #label种类个数

filename = sys.argv[1]
inputs_temp = []
labels_temp = []

file = open(filename) #读取训练数据
while 1:
    print count
    count += 1;
    line = file.readline()
    if not line:
        break
    linelist = line.strip().split();
    topwords = linelist[0]  #上屏词作为label
    pinyin = linelist[1] #输入串作为Input
    inputs_temp.append(coding(pinyin)) 
    if len(topwords) %2 != 0:
        print "Bad case",line
    label_len = len(topwords)/2 
    label_temp = []
    for i in range(int(label_len)):
        label_char = linelist[4+i] #读取上屏词在词表中的位置作为label
        label_temp.append(int(label_char))
    labels_temp.append(np.array(label_temp).astype(np.int64))
	

num_examples = count
splits_num = int(num_examples*0.9) #切分训练数据
num_epochs = int(sys.argv[2]) #迭代次数
test_num = int(sys.argv[3]) #测试样本数
num_batches_per_epoch = int(0.9*num_examples/batch_size)
num_batches_per_epoch_for_test = int(0.1*num_examples/batch_size) 
#inputs, labels = fake_data(num_examples, num_features, num_classes - 1)


inputs = np.asarray(inputs_temp)
labels = np.asarray(labels_temp)

# preprocess the input data 
train_inputs = inputs[:splits_num] #0.9 for train
test_inputs = inputs[:]  #0.1 for test
print "inputs\t",train_inputs.shape,train_inputs[0]

# preprocess the target data
train_targets = labels[:splits_num]
test_targets = labels[:]
print "targets\t",train_targets.shape,train_targets[0]


#THE MAIN TRAIN CODE

graph = tf.Graph()
with graph.as_default():
    # Has size [batch_size, max_stepsize, num_features], but the
    # batch_size and max_stepsize can vary along each step
    inputs = tf.placeholder(tf.float32, [None, None, num_features])
    #batch_size = inputs.get_shape
    #inputs = tf.transpose(inputs, [1, 0, 2])
    #inputs = tf.reshape(inputs, [-1, num_features])
    #inputs = tf.split(inputs, batch_size)

    #define dropout
    keep_prob = tf.placeholder(tf.float32)  

    # Here we use sparse_placeholder that will generate a
    # SparseTensor required by ctc_loss op.
    targets = tf.sparse_placeholder(tf.int32)

    # 1d array of size [batch_size]
    seq_len = tf.placeholder(tf.int32, [None])

    # Defining the cell
    # Can be:
    #   tf.nn.rnn_cell.RNNCell
    #   tf.nn.rnn_cell.GRUCell
    cell = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)

    ## Stacking rnn cells
    stack = tf.contrib.rnn.MultiRNNCell([cell] * num_layers,state_is_tuple=True)

    ## The second output is the last state and we will no use that
    outputs, _ = tf.nn.dynamic_rnn(stack, inputs, seq_len, dtype=tf.float32)
    #http://blog.csdn.net/u012436149/article/details/71080601
    #lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(num_hidden, forget_bias = 1.0)
    #lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(num_hidden, forget_bias = 1.0)

    #lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(lstm_fw_cell, output_keep_prob=keep_prob)
    #lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(lstm_bw_cell, output_keep_prob=keep_prob)
		#    
    #outputstuple, _= tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,lstm_bw_cell, 
		#                                                 inputs, seq_len,
    #                                                 dtype = tf.float32) 	

    #outputs = tf.concat(outputstuple, 2)
    shape = tf.shape(inputs)
    batch_s, max_timesteps = shape[0], shape[1]

    # Reshaping to apply the same weights over the timesteps
    outputs = tf.reshape(outputs, [-1, num_hidden])
    #outputs = outputstuple[-1]
    # Truncated normal with mean 0 and stdev=0.1
    # Tip: Try another initialization
    # see https://www.tensorflow.org/versions/r0.9/api_docs/python/contrib.layers.html#initializers
    with tf.name_scope("W"):
        W = tf.Variable(tf.truncated_normal([num_hidden,
                                         num_classes],
                                        stddev=0.1))
        tf.summary.histogram("/weights",W) #可视化观看变量
    # Zero initialization
    # Tip: Is tf.zeros_initializer the same?
    with tf.name_scope("b"):
        b = tf.Variable(tf.constant(0., shape=[num_classes]))
        tf.summary.histogram("/biases",b) #可视化观看变量

    # Doing the affine projection
    logits = tf.matmul(outputs, W) + b

    # Reshaping back to the original shape
    logits = tf.reshape(logits, [batch_s, -1, num_classes])

    # Time major
    logits = tf.transpose(logits, (1, 0, 2))
    
    #out = tf.Print(targets,data,message=None,first_n=None,summarize=None,name=None);
    #loss = tf.nn.ctc_loss(targets, logits, seq_len,preprocess_collapse_repeated=True, ctc_merge_repeated=True,ignore_longer_outputs_than_inputs=True)
    with tf.name_scope("loss"):
        loss = tf.nn.ctc_loss(targets, logits, seq_len,preprocess_collapse_repeated=True,ignore_longer_outputs_than_inputs=True)
        tf.summary.histogram("/loss",loss) #可视化观看变量
    cost = tf.reduce_mean(loss)

    optimizer = tf.train.MomentumOptimizer(initial_learning_rate,
                                           0.9).minimize(cost)

    # Option 2: tf.nn.ctc_beam_search_decoder
    # (it's slower but you'll get better results)
    decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, seq_len)
    decoded_beam, log_prob_beam = tf.nn.ctc_beam_search_decoder(logits, seq_len,top_paths=10)

    # Inaccuracy: label error rate
    with tf.name_scope("ler"):
        ler = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32),targets))
        tf.summary.histogram("/ler",ler) #可视化观看变量
    merged = tf.summary.merge_all()   
    writer = tf.summary.FileWriter("./",graph)
    saver = tf.train.Saver()

with tf.Session(graph=graph) as session:
    
    # Initializate the weights and biases
    #tf.global_variables_initializer().run()
    new_saver = tf.train.import_meta_graph('./14lstm/lstm-model.ckpt.meta')
    new_saver.restore(session,tf.train.latest_checkpoint('./14lstm'))

    #for curr_epoch in range(num_epochs):
    #    train_cost = train_ler = 0
    #    train_test_cost = train_test_ler = 0
    #    start = time.time()
    #    badcase = 0
    #    for batch in range(num_batches_per_epoch):

    #        # Getting the index
    #        indexes = [i % num_examples for i in range(batch * batch_size, (batch + 1) * batch_size)]
    #        #print "indexes",indexes
    #        batch_train_inputs = train_inputs[indexes]
    #        # Padding input to max_time_step of this batch
    #        batch_train_inputs, batch_train_seq_len = pad_sequences(batch_train_inputs)
    #        # Converting to sparse representation so as to to feed SparseTensor input
    #        batch_train_targets = sparse_tuple_from(train_targets[indexes])

    #        feed = {inputs: batch_train_inputs,
    #                targets: batch_train_targets,
    #                seq_len: batch_train_seq_len,
    #                keep_prob: 0.5
		#								}
    #       
    #        batch_cost, _ = session.run([cost, optimizer], feed)
    #        #if math.isinf(batch_cost):
    #        #    badcase += 1
    #        #    continue
    #        #print "batch_cost",batch_cost
    #        train_cost += batch_cost*batch_size
    #        train_ler += session.run(ler, feed_dict=feed)*batch_size

    #    for batch in range(num_batches_per_epoch_for_test):
    #        # Getting the index
    #        indexes = [i % int(num_examples*0.1) for i in range(batch * batch_size, (batch + 1) * batch_size)]
    #        #print "indexes",indexes
    #        batch_test_inputs = test_inputs[indexes]
    #        # Padding input to max_time_step of this batch
    #        batch_test_inputs, batch_test_seq_len = pad_sequences(batch_test_inputs)
    #        # Converting to sparse representation so as to to feed SparseTensor input
    #        batch_test_targets = sparse_tuple_from(test_targets[indexes])

    #        feed = {inputs: batch_test_inputs,
    #                targets: batch_test_targets,
    #                seq_len: batch_test_seq_len,
    #                keep_prob: 0.5}
    #       
    #        batch_test_cost = session.run(cost, feed)
    #        train_test_cost += batch_cost*batch_size
    #        train_test_ler += session.run(ler, feed_dict=feed)*batch_size
    #        
    #        if batch == (num_batches_per_epoch_for_test)-1:
    #            result = session.run(merged,feed) #run merged 
    #            writer.add_summary(result,curr_epoch) #result是summary类型的，需要放入writer中

    #    # Shuffle the data
    #    shuffled_indexes = np.random.permutation(splits_num)
    #    train_inputs = train_inputs[shuffled_indexes]
    #    train_targets = train_targets[shuffled_indexes]

    #    # Metrics mean
    #    train_test_cost /= (num_examples*0.1)
    #    train_test_ler /= (num_examples*0.1)

    #    log = "Epoch {}/{}, train_cost = {:.3f}, train_ler = {:.3f}, time = {:.3f}"
    #    print(log.format(curr_epoch+1, num_epochs, train_test_cost, train_test_ler, time.time() - start))
    #saver.save(session,'./lstm/lstm-model.ckpt')
    # Decoding all at once. Note that this isn't the best way

    #随机抽取指定个数的测试用例用于评估模型表现
    decode_index = np.asarray([np.randin() for i in range(test_num)])
    #decode_index = np.asarray([i for i in range(test_num)])
    # Padding input to max_time_step of this batch
    batch_train_inputs, batch_train_seq_len = pad_sequences(test_inputs[decode_index])
    #print test_inputs[decode_index[0]]

    # Converting to sparse representation so as to to feed SparseTensor input
    batch_train_targets = sparse_tuple_from(test_targets[decode_index])
    #print test_targets[decode_index[0]]

    feed = {inputs: batch_train_inputs,
            targets: batch_train_targets,
            seq_len: batch_train_seq_len,
            keep_prob: 0.5
            }

    # Decoding
    d_beam = session.run(decoded_beam, feed_dict=feed)
    d0 = d_beam[0]
    #d1 = d_beam[1]
    #d2 = d_beam[2]
    dense_decoded0 = tf.sparse_tensor_to_dense(d0, default_value=-1).eval(session=session)
    dictnory = {} 
    file = open("dict2NR") #读字典，用于解码结果可视化
    dictcount = 0
    while 1:
        #print dictcount
        dictcount += 1
        line = file.readline()
        if not line:
            break
        linelist = line.strip().split();
        if len(linelist) != 3:
            print line
            continue
        dictnory[int(linelist[2])] = linelist[0]

    for i, seq in enumerate(dense_decoded0):
        seq = [s for s in seq if s != -1]
        #第几个测试序列
        print('Sequence %d' % i)
        #原始序列
        print('\t Original:\t%s\t' % test_targets[decode_index][i]),
        print decode(test_inputs[decode_index][i]),"\t",
        for item in test_targets[decode_index][i]:
            print("%s\t" % dictnory[item]),
        print ""
        #解码序列
        #print('\t Decoded:\tTop1[\t%s' % seq),
        print '\t Decoded:\tTop\t1\t[\t',
        for item in seq:
            print("%s" % dictnory[item]),
        print "]"
        for t in range(1,10):
          d = d_beam[t]
          dense_decoded = tf.sparse_tensor_to_dense(d, default_value=-1).eval(session=session)
          print "\t\t\t\tTop",t+1,"[\t",
          for item in dense_decoded[i]:
              if item == -1:
                  print("%s" % "?"),
              else:
                  print("%s" % dictnory[item]),
          print "]"
    
