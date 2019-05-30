import json
from NeuralNet.Nets import SingleLayerTimeSeriesRNN as SRNN
from NeuralNet.Nets import SingleLayerTimeSeriesLSTM as SLSTM
from NeuralNet.Nets import MultipLayerTimeSeriesLSTM as MLSTM
import DataFileIO.DFIO as DataIO
import time

Configs = json.loads(open("config.json").read())
LookBack = Configs["LookBack"]

if __name__=='__main__':
	TrX, TrY, TeX, TeY = DataIO.DataRead(Configs["DataPath"])
	#model = MLSTM.build(TrX.shape[1], TrX.shape[2])
	model = SLSTM.build(TrX.shape[1], TrX.shape[2]) #LSTM模型
	#model = SRNN.build(TrX.shape[1], TrX.shape[2]) #RNN模型

	model.compile(loss=Configs["loss"], optimizer=Configs["optimizer"])
	StartTime = time.time()
	model.fit(TrX, TrY, epochs=Configs["epochs"], batch_size=Configs["batchsize"])
	EndTime = time.time()

	print("训练用时%.2f秒" %(EndTime-StartTime))
	model.save(Configs["ModelSaveFile"])
