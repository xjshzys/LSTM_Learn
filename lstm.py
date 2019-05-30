import pandas as pd
import numpy as np
import math
import json
from NeuralNet.Nets import SingleLayerTimeSeriesRNN as SRNN
from NeuralNet.Nets import SingleLayerTimeSeriesLSTM as SLSTM
from NeuralNet.Nets import MultipLayerTimeSeriesLSTM as MLSTM
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import datetime

Configs = json.loads(open("config.json").read())
LookBack = Configs["LookBack"]

def DataXY(Data):
	DataX, DataY = [], []
	for i in range(len(Data)-LookBack-1):
		temp = Data[i:(i+LookBack),0]
		DataX.append(temp)
		DataY.append(Data[LookBack+i,0])

	return np.array(DataX), np.array(DataY)

def DataRead():
	Data = pd.read_csv(Configs["DataPath"])
	Data = Data.values #转化格式

	#按比例分割训练集和测试集
	TrainDataLen = int(len(Data)*Configs["TrainSize"])
	TrainData = Data[0:TrainDataLen]
	TestData = Data[TrainDataLen:len(Data)]

	TrainX, TrainY = DataXY(TrainData)
	TestX, TestY = DataXY(TestData)

	TrainX = TrainX.reshape(len(TrainX), LookBack, 1) #单特征预测，LookBack个数据作为多个特征被观测而不是整体观测
	TestX = TestX.reshape(len(TestX), LookBack, 1)

	#print(TrainX.shape, TestX.shape)

	return TrainX, TrainY, TestX, TestY

if __name__=='__main__':
	NumReal, NumPred = [], []

	#将前LookBack个数据补零
	for i in range(LookBack):
		NumPred.append(0)

	TrX, TrY, TeX, TeY = DataRead()
	#model = MLSTM.build(TrX.shape[1], TrX.shape[2])
	model = SLSTM.build(TrX.shape[1], TrX.shape[2]) #LSTM模型
	#model = SRNN.build(TrX.shape[1], TrX.shape[2]) #RNN模型

	model.compile(loss=Configs["loss"], optimizer=Configs["optimizer"])
	StartTime = datetime.datetime.now()
	model.fit(TrX, TrY, epochs=Configs["epochs"], batch_size=Configs["batchsize"])
	EndTime = datetime.datetime.now()

	model.save(Configs["ModelSaveFile"])

	for i in TeX:
		temp=[]
		for j in i:
			temp.append(j)
		x = np.reshape(temp, (1, len(temp), 1)) #整理需要预测的数据成模型需要的形状
		TestPredict = model.predict(x)
		NumReal.append(temp[0]) #只添加第一个数据
		NumPred.append(TestPredict[0])
		#print(temp[0], TestPredict)

	#均方根误差计算，位移LookBack个步长
	trainScore = math.sqrt(mean_squared_error(NumReal[LookBack:], NumPred[LookBack:LookBack*-1]))

	#画图
	plt.rcParams['font.sans-serif'] = ['KaiTi']
	plt.rcParams['axes.unicode_minus'] = False
	plt.rcParams['font.size'] = 8

	plt.figure(num=1, figsize=(15, 8),dpi=150)
	plt.title("用时 %.2f 秒, 损失值为 %.5f" %((EndTime-StartTime).microseconds/10000, trainScore))
	plt.plot(NumReal, label='真实值')
	plt.plot(NumPred, label='预测值')
	plt.legend()
	plt.show()
