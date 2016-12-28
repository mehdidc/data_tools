from scipy.io import loadmat
import numpy as np
X = loadmat('usps_all.mat')
X = X['data']
print(X.shape)
X = X.transpose((1, 2, 0))
X = X.reshape((1100, 10, 1, 16, 16))
X = X.transpose((0, 1, 2, 4, 3))
X = X.reshape((11000, 1, 16 , 16))
y = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0] * 1100
y = np.array(y)
np.savez('data.npz', X=X, y=y)
print(X.shape, y.shape)
