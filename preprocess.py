#!/usr/bin/env python

# read 10 years of game records in gtp from KGS
# convert to int array of sequences
# pad to matrix
# save to an hdf5 file

import glob
import h5py
from keras.utils.io_utils import HDF5Matrix
from keras.preprocessing import sequence
from moves import MoveEncoder

alphabet = MoveEncoder('moves.gtp')
max_features = alphabet.size()

# train only on the first 100 moves
# reduce maxlen if you run into out-of-memory problems
maxlen = 100 

X = []

for year in range(2006, 2016):
	print("processing year {0}".format(year))
	for filename in glob.glob("./data/kgs-19-{0}/*.gtp".format(year)):
		with open(filename) as game:
			lines = list(game)
			
			n = len(lines)
			if n > maxlen:
				n = maxlen
			# starts after 'boardsize 19', 'clear_board', 'komi'
			X.append([alphabet.encode(c) for c in lines[0:n]])

X = sequence.pad_sequences(X, maxlen=maxlen)

f = h5py.File('2006-2015-{0}.h5'.format(maxlen), 'w')
X_dset = f.create_dataset('data', X.shape, dtype='i')
X_dset[:] = X
f.close()

