# ctc-distributed-parallel-strategy.
用ctc模型实现简单的拼音输入法，可以实现输入串直接向汉字的转化。
使用进行分布式同步训练。
为了满足ctc条件，对原始输入的每两个字符之间增加“*”以增加输入串长度
结果如下所示：
