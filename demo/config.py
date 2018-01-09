import os
# data config
data_path = "/search/data/jixingguang/120W_py_label"
#data_path = "/search/data/jixingguang/1W_py_label"
dataset = "ctc_test"
log_dir = "./model/log"
#model_dir = "./model"
model_dir = os.path.join(data_path, "model")
model_name = "dbg_ctc"
#dictdir = "./simple_dict"
dictdir = "py_list.pyseg.dic"

# model config
num_layer = 1
embed_size = 1024
#layer_size = [1024, 1024]
layer_size = 1024
num_features = 27
num_hidden = 1024
#num_classes = 5002
num_classes = 441
epcho_num_for_test  = 1

# train config
vocab_size = 10000
batch_size = 1024
#batch_size = 128
num_step = 64
step_per_log = 100
top_k = 3
lstm_keep_prob = 0.5
input_keep_prob = 0.5
output_keep_prob = 0.5
decay_rate = 0.8
max_grad_norm = 5.0
case_num_4_test = 20


