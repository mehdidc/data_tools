import h5py
import numpy as np
data = h5py.File('fonts.hdf5', 'r')
X = data['fonts']
f = h5py.File("data_shuffled.hdf5", "w")
ind = np.arange(X.shape[0] * X.shape[1])
np.random.shuffle(ind)
x = f.create_dataset("X", (X.shape[0]*X.shape[1], 1, 64, 64), dtype='f')
y = f.create_dataset("y", (X.shape[0]*X.shape[1],), dtype='i')
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        x[ind[i * X.shape[1] + j], 0, :, :] = X[i, j]
        y[ind[i * X.shape[1] + j]] = j
