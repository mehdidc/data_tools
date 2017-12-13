import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.utils import shuffle
from skimage.transform import resize

data = fetch_olivetti_faces()
X = data['images']
Xr = np.empty((X.shape[0], 32, 32))
for i, x in enumerate(X):
    x = resize(x, (32, 32), preserve_range=True)
    Xr[i] = x
X = Xr
y = data['target']
X = X[:, np.newaxis, :, :]
print(X.shape, y.shape)
print(X.min(), X.max())
X *= 255.
X, y = shuffle(X, y, random_state=32)

print(X.min(), X.max())
np.savez('data.npz', X=X, y=y)
