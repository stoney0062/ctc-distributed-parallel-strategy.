# -*- coding:utf-8 -*-
import matplotlib
import numpy as np
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import sys
import os


def PlotFigure(ylabel):
# 颜色列表
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

	# 设置横坐标和纵坐标的名称
	plt.xlabel("Epoch")
	plt.ylabel(ylabel)
	# 图的标题
	#plt.title('9/26/all shu ru liang tong ji')
	# 第一根线的纵坐标
	threadList1 = [y for y in range(len(dataList1))];
	# 根据横坐标和纵坐标画第一根线
	line1, = plt.plot(threadList1, dataList1)
	# 设置线的颜色宽度等
	plt.setp(line1, color=colorList[0], linewidth=0.5)

	picname = os.path.join("./",ylabel+".png")
	plt.savefig(picname, dpi=120)

def main():
	PlotFigure(sys.argv[1])

if __name__ == '__main__':
	main()
