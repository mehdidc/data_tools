from lasagnekit.datasets.stl import STL
import numpy as np
from skimage.transform import resize
data = STL('unlabeled')
data.load()
X = data.X
x = np.empty((100000, 32, 32, 3))
for i in range(X.shape[0]):
    x[i] = resize(X[i], (32, 32), preserve_range=True)
    if i % 100 == 0:
        print(i)
x = x.transpose((0, 3, 1, 2))
np.savez('train.npz', x=x)
