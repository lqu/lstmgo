#!/usr/bin/env python

import glob
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding, LSTM
from keras.callbacks import ModelCheckpoint
from keras.metrics import top_k_categorical_accuracy
from keras.utils.io_utils import HDF5Matrix

############################################
#	build model
############################################
model = Sequential()
model.add(Embedding(735, 512))
model.add(LSTM(512, return_sequences=True))
model.add(LSTM(512))
model.add(Dense(735))
model.add(Activation('softmax'))

def inTopN(n):
	return lambda y1, y2: top_k_categorical_accuracy(y1, y2, k=n)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy', inTopN(5), inTopN(10)])
model.load_weights('weights-100-moves-3.1208-3.2425.hdf5')

###########################################
#	load data
###########################################

# train only on the first 100 moves
# reduce maxlen if you run into out-of-memory problems
maxlen = 100

# put aside 4096 games for validation
records = HDF5Matrix('all-100.h5', 'data', start=0, end=4096)
x_val = []
y_val = []

for rec in records:
	for k in range(3, maxlen):
		x_val.append(rec[0:k])
		y_val.append(rec[k])

XV = sequence.pad_sequences(x_val, maxlen=maxlen)
YV = np_utils.to_categorical(y_val, nb_classes=735)

# number of records
N = 140116
i = 4096
batch_size = 4096


while i < N:
	start = i
	end = i + batch_size
	if end > N:
		end = N
	records = HDF5Matrix('all-100.h5', 'data', start=start, end=end)

	x_train = []
	y_train = []

	for rec in records:
		for k in range(3, maxlen):
			x_train.append(rec[0:k])
			y_train.append(rec[k])

	X = sequence.pad_sequences(x_train, maxlen=maxlen)
	Y = np_utils.to_categorical(y_train, nb_classes=735)


	filepath="no-embedding-100-moves-{loss:.4f}-{val_loss:.4f}.hdf5"
	checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
	callbacks_list = [checkpoint]
	# fit the model
	#model.fit(X, Y, nb_epoch=1, batch_size=256, callbacks=callbacks_list, validation_split=0.05)
	model.fit(X, Y, nb_epoch=1, batch_size=256, callbacks=callbacks_list, validation_data=(XV, YV))
	#model.evaluate(XV, YV)

	i = end
