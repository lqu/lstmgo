#!/usr/bin/env python

import glob
import numpy
from keras.preprocessing import sequence
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding, LSTM
from keras.metrics import top_k_categorical_accuracy
from moves import MoveEncoder

alphabet = MoveEncoder('moves.gtp')

# train only on the first 50 moves
# reduce maxlen if you run into out-of-memory problems
maxlen = 50

model = Sequential()
model.add(LSTM(output_dim=735, input_length=50, input_dim=735))
model.add(Dense(output_dim=735))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')
model.load_weights('t6-loss-2.4902-val_loss-2.6000-.h5')

x_int = [[0, 1, 10]]
x_int = sequence.pad_sequences(x_int, maxlen=maxlen)
X = numpy.array([to_categorical(i, 735) for i in x_int])

for i in range(50):
	Y = model.predict(X)
	index = numpy.argmax(Y[0])
	print alphabet.decode(index)
	new_state = numpy.append(x_int[0], index)[1:]
	x_int[0] = new_state
	X = numpy.array([to_categorical(i, 735) for i in x_int])
