#!/usr/bin/env python

import glob
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding, LSTM
from keras.callbacks import ModelCheckpoint
from keras.metrics import top_k_categorical_accuracy
from moves import MoveEncoder

alphabet = MoveEncoder('moves.gtp')
max_features = alphabet.size()

# train only on the first 50 moves
# reduce maxlen if you run into out-of-memory problems
maxlen = 50

x_train = []
y_train = []

testfiles = 'data/kgs-19-2011/*.gtp'

for filename in glob.glob(testfiles):
	with open(filename) as game:
		lines = list(game)
		
		n = len(lines)
		if n > maxlen:
			n = maxlen
		# starts after 'boardsize 19', 'clear_board', 'komi', 'fixed_handicap'
		for i in range(4, n):
			x_train.append([alphabet.encode(c) for c in lines[0:i]])
			y_train.append(alphabet.encode(lines[i]))

X = sequence.pad_sequences(x_train, maxlen=maxlen)
Y = np_utils.to_categorical(y_train, nb_classes=735)

model = Sequential()
model.add(Embedding(735, 735))
model.add(LSTM(735))
model.add(Dense(735))
model.add(Activation('softmax'))

def inTopN(n):
	return lambda y1, y2: top_k_categorical_accuracy(y1, y2, k=n)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy', inTopN(5), inTopN(10)])

filepath="weights-50-moves-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
# fit the model
model.fit(X, Y, nb_epoch=10, batch_size=256, callbacks=callbacks_list, validation_split=0.1)
