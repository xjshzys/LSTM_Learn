import pandas as pd
import numpy as np
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import SimpleRNN
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

DataPath = "Data\\"
DataFile = "sinewave.csv"

TrainSize = 0.6 #训练集占比
LookBack = 10 #观测步长

def BuildModel(ISX, ISY):
	model = Sequential()

	model.add(LSTM(32,input_shape=(ISX,ISY)))
	model.add(Dense(1))

	return model

def BuildSimRnn(ISX, ISY):
	model = Sequential()

	model.add(SimpleRNN(32,input_shape=(ISX,ISY)))
	model.add(Dense(1))

	return model

def DataXY(Data):
	DataX, DataY = [], []
	for i in range(len(Data)-LookBack-1):
		temp = Data[i:(i+LookBack),0]
		DataX.append(temp)
		DataY.append(Data[LookBack+i,0])
	
	return np.array(DataX), np.array(DataY)

def DataRead():
	Data = pd.read_csv(DataPath+DataFile)
	Data = Data.values #转化格式

	#按比例分割训练集和测试集
	TrainDataLen = int(len(Data)*TrainSize)
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
	#model = BuildModel(TrX.shape[1], TrX.shape[2]) #LSTM模型
	model = BuildSimRnn(TrX.shape[1], TrX.shape[2]) #RNN模型
	
	model.compile(loss='mean_squared_error', optimizer='adam')
	model.fit(TrX, TrY, epochs=1, batch_size=1)
 
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
	print('Train Score: %.2f RMSE' % (trainScore))

	#画图
	fig = plt.figure(num=1, figsize=(15, 8),dpi=80) 
	plt.plot(NumReal, label='Real')
	plt.plot(NumPred, label='Predict')
	plt.show()

	#print(TeX, TrainPredict)