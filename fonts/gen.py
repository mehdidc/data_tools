import h5py
import numpy as np
X, y = np.load('ds_all_64.npy')
letters = 'abcdefghijklmnopqrstuvwxyz'
y = map(lambda s:s[s.index('ttf'):].split('-')[1], y)
y = map(lambda s:letters.index(s), y)
X = np.array(X.tolist())
y = np.array(y)
X = X.reshape((X.shape[0], 1, 64, 64))
np.savez('fonts.npz', X=X, y=y)
#f = h5py.File('fonts.hdf5')
#print(y)
#f['X'] = X
#f['y'] = y
