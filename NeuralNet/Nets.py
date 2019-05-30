from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import SimpleRNN

class SingleLayerTimeSeriesRNN:
	@staticmethod
	def build(ISX, ISY):
		model = Sequential()

		model.add(SimpleRNN(32,input_shape=(ISX,ISY)))
		model.add(Dense(1))

		return model

class SingleLayerTimeSeriesLSTM:
	@staticmethod
	def build(ISX, ISY):
		model = Sequential()

		model.add(LSTM(32,input_shape=(ISX,ISY)))
		model.add(Dense(1))

		return model

class MultipLayerTimeSeriesLSTM:
	@staticmethod
	def build(ISX, ISY):
		model = Sequential()

		model.add(LSTM(32, input_shape=(ISX,ISY), return_sequences=True))
		model.add(LSTM(64, return_sequences=True))
		model.add(LSTM(128))

		model.add(Dense(1))

		return model