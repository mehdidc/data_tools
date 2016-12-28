from datakit.cifar import load
import numpy as np
data = load()

X_train = data['train']['X']
y_train = data['train']['y']

print(X_train.shape)
np.savez('train.npz', X=X_train, y=y_train)

X_test = data['test']['X']
y_test = data['test']['y']


np.savez('test.npz', X=X_test, y=y_test)
