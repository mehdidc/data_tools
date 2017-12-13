import sys
import numpy as np
from datakit.mnist import load
from skimage.transform import resize
from sklearn.utils import shuffle

def resize_set(x, w, h, **kw):
    x_out = np.empty((x.shape[0], 1, w, h))
    for i in range(len(x)):
        x_out[i, 0] = resize(x[i, 0], (w, h), **kw)
    return x_out.astype(np.float32)

digits_train = np.load('../mnist/train.npz')
digits_test = np.load('../mnist/test.npz')


letters = np.load('fonts.npz')

Xdtrain, ydtrain = digits_train['X'], digits_train['y']
Xdtest, ydtest = digits_test['X'], digits_test['y']
Xd = np.concatenate((Xdtrain, Xdtest), axis=0)
yd = np.concatenate((ydtrain, ydtest), axis=0)

Xl, yl = letters['X'], letters['y']
Xl = 255 - Xl
Xl = resize_set(Xl, 28, 28, preserve_range=True)

X = np.concatenate((Xd, Xl), axis=0)
y = np.concatenate((yd, yl + 10), axis=0)
X, y = shuffle(X, y, random_state=42)
np.savez('digits_and_letters.npz', X=X, y=y)
