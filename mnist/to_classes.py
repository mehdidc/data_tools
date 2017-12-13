import numpy as np
from skimage.io import imsave
import os
try:
    os.makedirs('img_classes/0/imgs')
    os.makedirs('img_classes/1/imgs')
    os.makedirs('img_classes/2/imgs')
    os.makedirs('img_classes/3/imgs')
    os.makedirs('img_classes/4/imgs')
    os.makedirs('img_classes/5/imgs')
    os.makedirs('img_classes/6/imgs')
    os.makedirs('img_classes/7/imgs')
    os.makedirs('img_classes/8/imgs')
    os.makedirs('img_classes/9/imgs')
except Exception:
    pass

data = np.load('train.npz')

for i, (x, y) in enumerate(zip(data['X'], data['y'])):
    x = x[0]
    imsave('img_classes/{}/{}.png'.format(y, i), x)
