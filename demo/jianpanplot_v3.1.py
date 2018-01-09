# -*- coding:utf-8 -*-
import matplotlib
import numpy as np
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import sys
import os


def PlotFigure(ylabel):
# ��ɫ�б�
	colorList = ['b','g','r','c','m','y','k']
	dataList1 = []
	for line in sys.stdin:
		if not line:
			break;
		try:
			linesList = float(line.strip());
		except:
			continue
		dataList1.append(linesList)

	# ���ú�����������������
	plt.xlabel("Epoch")
	plt.ylabel(ylabel)
	# ͼ�ı���
	#plt.title('9/26/all shu ru liang tong ji')
	# ��һ���ߵ�������
	threadList1 = [y for y in range(len(dataList1))];
	# ���ݺ�����������껭��һ����
	line1, = plt.plot(threadList1, dataList1)
	# �����ߵ���ɫ��ȵ�
	plt.setp(line1, color=colorList[0], linewidth=0.5)

	picname = os.path.join("./",ylabel+".png")
	plt.savefig(picname, dpi=120)

def main():
	PlotFigure(sys.argv[1])

if __name__ == '__main__':
	main()
