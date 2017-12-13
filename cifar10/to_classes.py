import numpy as np
from skimage.io import imsave
import os
nb_classes = 100
for i in range(nb_classes):
    try:
        os.makedirs('img_classes/{}/imgs'.format(i))
    except Exception:
        pass

data = np.load('train.npz')

for i, (x, y) in enumerate(zip(data['X'], data['y'])):
    x = x.transpose((1, 2, 0))
    imsave('img_classes/{}/{}.png'.format(y, i), x)
