import pandas as pd
import numpy as np
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error

DataPath = "Data\\"
DataFile = "Numarray.csv"

TrainSize = 0.6 #训练集占比
LookBack = 3 #观测步长

def BuildModel():
	model = Sequential()

	model.add(LSTM(32,input_shape=(1,LookBack)))
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
	Data = Data.values

	TrainDataLen = int(len(Data)*TrainSize)
	TrainData = Data[0:TrainDataLen]
	TestData = Data[TrainDataLen:len(Data)]

	TrainX, TrainY = DataXY(TrainData)
	TestX, TestY = DataXY(TestData)

	TrainX = TrainX.reshape(TrainX.shape[0], 1, TrainX.shape[1])
	TestX = TestX.reshape(TestX.shape[0], 1, TestX.shape[1])

	print(TrainX.shape[0],TrainX.shape[1])

	return TrainX, TrainY, TestX, TestY

if __name__=='__main__':
	TrX, TrY, TeX, TeY = DataRead()
	model = BuildModel()
	
	model.compile(loss='mean_squared_error', optimizer='adam')
	model.fit(TrX, TrY, epochs=3, batch_size=3, verbose=2)

	TrainPredict = model.predict(TrX)

	print(TrX, TrainPredict)
	
	#trainScore = math.sqrt(mean_squared_error(TrX[0], TrainPredict[:,0])) 
	#print('Train Score: %.2f RMSE' % (trainScore))