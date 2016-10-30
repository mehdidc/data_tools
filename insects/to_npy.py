
import numpy as np
from skimage.io import imread

labels = open("labels", "r").readlines()
y = np.array([int(l) for l in labels])

nb = y.shape[0]
X = np.zeros( (nb, 64, 64, 3) )

for i in xrange(nb):
    filename = "%i.jpg" % (i,)
    image = imread(filename)
    X[i] = image

np.savez("data.npy", X=X, y=y)
