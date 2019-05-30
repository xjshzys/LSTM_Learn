import pandas as pd
import numpy as np

def CovData2XY(Data, LookBack):
	DataX, DataY = [], []
	for i in range(len(Data)-LookBack-1):
		temp = Data[i:(i+LookBack),0]
		DataX.append(temp)
		DataY.append(Data[LookBack+i,0])

	return np.array(DataX), np.array(DataY)

def DataRead(FilePath, SplitSize=0.6, LookBack=10):
	"""
	读取数据

	FilePath = 数据文件所在位置

	SplitSize = 训练集分割比例

	LookBack = 预测步长

	返回训练集XY形式，测试集XY形式
	"""
	Data = pd.read_csv(FilePath)
	Data = Data.values #转化格式

	#按比例分割训练集和测试集
	TrainDataLen = int(len(Data)*SplitSize)
	TrainData = Data[0:TrainDataLen]
	TestData = Data[TrainDataLen:len(Data)]

	TrainX, TrainY = CovData2XY(TrainData, LookBack)
	TestX, TestY = CovData2XY(TestData, LookBack)

	TrainX = TrainX.reshape(len(TrainX), LookBack, 1) #单特征预测，LookBack个数据作为多个特征被观测而不是整体观测
	TestX = TestX.reshape(len(TestX), LookBack, 1)

	return TrainX, TrainY, TestX