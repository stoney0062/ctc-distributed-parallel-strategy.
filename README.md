# ctc-distributed-parallel-strategy.
用ctc模型实现简单的拼音输入法，可以实现输入串直接向汉字的转化。
使用进行分布式同步训练。
解码过程使用beam_search，取Top10
为了满足ctc条件，对原始输入的每两个字符之间增加“*”以增加输入串长度
结果如下所示：
Sequence	0
	Epcho 60 Origin:	[ 88 125   0   0   0] 	*e*r*h* 	er	hao	
	Epcho 60 Decoded: 			Top 0 	er 	hao 	
			Top 1 	er 	huo 	
			Top 2 	er 	ha 	
			Top 3 	er 	huan 	
			Top 4 	e 	er 	hao 	
			Top 5 	e 	er 	huo 	
			Top 6 	e 	er 	ha 	
			Top 7 	er 	hu 	
			Top 8 	er 	hao 	
			Top 9 	e 	er 	huan 	

Sequence	1
	Epcho 60 Origin:	[80  0  0  0  0] 	*d*u*a*n* 	duan	
	Epcho 60 Decoded: 			Top 0 	da 	
			Top 1 	dou 	
			Top 2 	di 	
			Top 3 	dian 	
			Top 4 	deng 	
			Top 5 	dao 	
			Top 6 	duo 	
			Top 7 	dai 	
			Top 8 	dui 	
			Top 9 	dong 	

Sequence	2
	Epcho 60 Origin:	[265 119   0   0   0] 	*p*i*n*g*g*u*o* 	ping	guo	
	Epcho 60 Decoded: 			Top 0 	ping 	guo 	
			Top 1 	pei 	guo 	
			Top 2 	pi 	guo 	
			Top 3 	pai 	guo 	
			Top 4 	pao 	guo 	
			Top 5 	pian 	guo 	
			Top 6 	pa 	guo 	
			Top 7 	peng 	guo 	
			Top 8 	pu 	guo 	
			Top 9 	pang 	guo 	

Sequence	3
	Epcho 60 Origin:	[313 143   0   0   0] 	*s*h*a*n*g*j*i*a* 	shang	jia	
	Epcho 60 Decoded: 			Top 0 	shi 	jie 	
			Top 1 	shi 	jia 	
			Top 2 	shi 	ju 	
			Top 3 	shi 	jin 	
			Top 4 	shen 	jie 	
			Top 5 	shen 	jia 	
			Top 6 	shang 	jie 	
			Top 7 	shang 	jia 	
			Top 8 	shou 	jie 	
			Top 9 	shou 	jia 	
