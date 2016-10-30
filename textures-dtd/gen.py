import os
from glob import glob
from skimage.io import imread
from skimage.transform import resize
import numpy as np

X = []
y = []
for dirname in glob('images/*'):
    category_name = dirname[len('images/'):]
    images = list(glob('images/{0}/*.jpg'.format(category_name)))
    for image in images:
        image = imread(image)
        image = resize(image, (200, 200), preserve_range=True)
        X.append(image)
        y.append(category_name)

X = np.array(X)
y = np.array(y)
print(X.shape, y.shape)
np.savez("output", X=X, y=y)
