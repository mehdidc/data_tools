from skimage.io import imsave
import numpy as np
from machinedesign.viz import grid_of_images_default
data = np.load('data.npz')
X = data['X']
y = data['y']
np.random.shuffle(X)
img = grid_of_images_default(255 - X[0:256])
imsave('samples.png', img)
