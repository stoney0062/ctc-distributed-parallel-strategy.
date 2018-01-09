# -*- coding: utf-8 -*
from __future__ import absolute_import
from __future__ import division
#from __future__ import print_function

import os
import time
import math
import numpy as np
from collections import Counter, defaultdict
from itertools import takewhile, repeat
import multiprocessing
import pickle
import pdb

from utils import sparse_tuple_from as sparse_tuple_from
from utils import pad_sequences as pad_sequences
from utils import input_coding as coding
from utils import input_decode as decode



class Reader(object):
	def __init__(self, data_dir, vocab_size):
		self.BOS = "<s>"
		self.EOS = "</s>"
		self.UNK = "<unk>"
		self.data_dir = data_dir
		self.vocab_size = vocab_size

		#self.build_vocab()
	
	def build_vocab(self):
		vocab_path = os.path.join(self.data_dir, "vocabulary.pkl")
		train_dir = os.path.join(self.data_dir, "train")
		valid_dir = os.path.join(self.data_dir, "valid")
		test_dir = os.path.join(self.data_dir, "test")
		
		if os.path.isfile(vocab_path):
			#self.words = open(vocab_path).read().replace("\n", " ").split()
			vocab_file = open(vocab_path, 'rb')
			self.words = pickle.load(vocab_file)
			vocab_file.close()
		else:
			data = []
			train_files = os.listdir(train_dir)
			for f in [os.path.join(train_dir, train_file) for train_file in train_files]:
				data.extend(self.read_words(f))
			valid_files = os.listdir(valid_dir)
			for f in [os.path.join(valid_dir, valid_file) for valid_file in valid_files]:
				data.extend(self.read_words(f))
			test_files = os.listdir(test_dir)
			for f in [os.path.join(test_dir, test_file) for test_file in test_files]:
				data.extend(self.read_words(f))
			
			counter = Counter(data)  # sort words '.': 5, ',': 4......
			count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
			words = list(zip(*count_pairs)[0])
			print("total vocab size: %d"%len(words))
			
			# make sure <unk> is in vocabulary
			if self.UNK not in words:
				words.insert(0, self.UNK)
			# make sure EOS is id 0
			words.insert(0, self.EOS)
			self.words = words[:self.vocab_size]
			assert len(self.words) == self.vocab_size

			# Save the vocabulary with pickle for future use
			vocab_file = open(vocab_path, 'wb')
			pickle.dump(self.words, vocab_file)
			vocab_file.close()
	
		print("vocab size: %d"%len(self.words))
		self.word2id = dict(zip(self.words, range(self.vocab_size))) 
	
	def read_words(self, file_path):  # return 1-D list
		words = []
		with open(file_path) as f:
			for line in f:
				words.extend(line.strip().split())
		return words

	def from_line_to_id(self, line):
		word_list = line.strip().split()
		word_list.append(self.EOS)
		id_list = [self.word2id[word] for word in word_list]
		return id_list

	def get_batch_from_file(self, data_file, batch_size, num_step):
		raw_data = []
		with open(data_file, 'r') as f:
			for line in f:
				id_list = self.from_line_to_id(line)
				raw_data.extend(id_list)
		batch_len = len(raw_data) // batch_size
		batch_data = np.array(raw_data[:batch_len * batch_size]).reshape([batch_size, batch_len])
		batch_num = (batch_len-1) // num_step
		for i in range(batch_num):
			batch_input = batch_data[:, i*num_step:(i+1)*num_step]
			batch_output = batch_data[:, i*num_step+1:(i+1)*num_step+1]
			yield batch_input, batch_output
			
	def get_batch_from_file_ctc(self, data_file, batch_size):
		count = 0
		labels_temp = []
		inputs_temp = []
		file = open(data_file) #读取训练数据
		while 1:
			line = file.readline()
			#if count%10000 == 0:
			#		print count,line
			if not line:
					break
			if "NONE" in line:
					continue
			linelist = line.strip().split();
			topwords = linelist[0]  #上屏词作为label
			pinyin = linelist[1] #输入串作为Input
			if len(topwords) %2 != 0:
					#print "Bad case",line
					continue
			label_len = len(topwords)/2 
			label_temp = []
			#if int(label_len)+2 > len(linelist):
			#		continue
			for i in range(int(label_len)):
					#label_char = linelist[2+i] #读取上屏词在词表中的位置作为label
					label_char = linelist[3+int(label_len)+i] #读取上屏词在词表中的位置作为label
					label_temp.append(int(label_char))
			if len(label_temp) == 0:
						continue
			count += 1;
			inputs_temp.append(coding(pinyin))
			labels_temp.append(np.array(label_temp).astype(np.int64))
			#inputs = np.asarray(coding(pinyin))	
			#labels = np.asarray(np.array(label_temp).astype(np.int64))
		num_examples = count
		num_batches_per_epoch = num_examples//batch_size
		inputs = np.asarray(inputs_temp)
		labels = np.asarray(labels_temp)
		for batch in range(num_batches_per_epoch):
			#Getting the index
			indexes = [i % num_examples for i in range(batch * batch_size, (batch + 1) * batch_size)]
			yield inputs[indexes],labels[indexes]

	def get_dictnory(self,dictdir):
		dictnory = {}  
		file = open("dictdir") #读字典，用于解码结果可视化
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
		return dictnory
	
