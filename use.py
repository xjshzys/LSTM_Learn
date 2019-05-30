import json
import math

import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from sklearn.metrics import mean_squared_error

import DataFileIO.DFIO as DataIO

Configs = json.loads(open("config.json").read())
LookBack = Configs["LookBack"]

if __name__ == "__main__":
	NumReal, NumPred = [], []
	for i in range(LookBack):
		NumPred.append(0)

	print("正在读取模型")
	model = load_model(Configs["ModelSaveFile"])

	print("正在读取数据")
	TrX, TrY, Test = DataIO.DataRead(Configs["DataPath"])

	print("正在预测")
	for i in Test:
		x = np.reshape(i, (1, LookBack, 1)) #整理需要预测的数据成模型需要的形状
		TestPredict = model.predict(x)
		NumReal.append(i[0]) #只添加第一个数据
		NumPred.append(TestPredict[0])

	#均方根误差计算，位移LookBack个步长
	trainScore = math.sqrt(mean_squared_error(NumReal[LookBack:], NumPred[LookBack:LookBack*-1]))

	#画图
	plt.rcParams['font.sans-serif'] = ['KaiTi']
	plt.rcParams['axes.unicode_minus'] = False
	plt.rcParams['font.size'] = 8

	plt.figure(num=1, figsize=(15, 8),dpi=150)
	plt.title("损失值为 %.5f" %(trainScore))
	plt.plot(NumReal[LookBack:], label='真实值')
	plt.plot(NumPred[LookBack:], label='预测值')
	plt.legend()
	plt.show()
