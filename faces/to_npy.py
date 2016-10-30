from pathlib import Path
from skimage import io,  color
import numpy as np

nb_batches = 10
shape = None
size = 0
g = '**/*.jpg'
paths = list(Path('.').glob(g))

for file_path in paths:
    if shape is None:
        img = io.imread(str(file_path))
        shape = img.shape
    size += 1

b = 0
batch_size = size / nb_batches
for first in range(0, len(paths), batch_size):
    last = min(first + batch_size, len(paths))    
    b_size = last - first + 1
    imgs_grayscale = np.empty((b_size, shape[0], shape[1]))
    #imgs_color = np.empty((b_size, shape[0], shape[1], shape[2]))
    imgs_color = None
    for i, p in enumerate(paths[first:last]):
        img = io.imread(str(p))
        #imgs_color[i] = img
        imgs_grayscale[i] = color.rgb2gray(img)
    np.savez("data_batch_%d.npy" % (b,), X_grayscale=imgs_grayscale, X_rgb=imgs_color)
    b += 1
