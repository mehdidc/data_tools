import sys
import numpy as np
from datakit.mnist import load
from skimage.transform import resize
def resize_set(x, w, h, **kw):
    x_out = np.empty((x.shape[0], 1, w, h))
    for i in range(len(x)):
        x_out[i, 0] = resize(x[i, 0], (w, h), **kw)
    return x_out.astype(np.float32)
data = load(which='all')
d=  {'X': [], 'y': []}
d['X'].append(data['train']['X'])
d['y'].append(data['train']['y'][:, 0])
d['X'].append(data['test']['X'])
d['y'].append(data['test']['y'][:, 0])
data = np.load('fonts.npz')
X = 255 - data['X']
X = resize_set(X, 28, 28, preserve_range=True)
y = data['y'] + 10
for _ in range(2):
    d['X'].append(X)
    d['y'].append(y)
ind = np.arange(len(X))
nb = 70000 - len(X) * 2
np.random.shuffle(ind)
ind = ind[0:nb]
d['X'].append(X[ind])
d['y'].append(y[ind])
print(d['y'][0].shape)
print(d['y'][1].shape)
print(d['y'][2].shape)
print(d['y'][3].shape)
print(d['y'][4].shape)
X = np.concatenate(d['X'], axis=0)
y = np.concatenate(d['y'], axis=0)
print(X.shape)
print(X.min(), X.max())
print(y.max())
np.savez('fonts_and_digits.npz', X=X, y=y)
