from skimage.io import imsave
import numpy as np
import os

from sklearn.model_selection import train_test_split


def to_folder(X, y, folder):
    try:
        os.mkdir('{}/{}'.format(folder, 'images'))
    except Exception:
        pass
    for idx, (x, y) in enumerate(zip(X, y)):
        x = x[0]
        x = 1 - x
        imsave('{}/images/{}.png'.format(folder, idx), x)

data = np.load('data.npz')
X, y = data['X'], data['y']

X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, test_size=0.1, random_state=42)

print(X_train.shape, y_train.shape)
print(X_valid.shape, y_valid.shape)
print(X_test.shape, y_test.shape)

to_folder(X_train, y_train, 'train')
to_folder(X_valid, y_valid, 'valid')
to_folder(X_test, y_test, 'test')
