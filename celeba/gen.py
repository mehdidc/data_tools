import numpy as np
from skimage.io import imread
from skimage.transform import resize
import glob

fd = open('list_attr_celeba.txt')
fd.readline()
fd.readline()

X = []
y = []
for i in range(1, 83812):
    filename = 'img_align_celeba_png/{:06d}.png'.format(i)
    img = imread(filename)
    img = resize(img, (64, 64))
    img = img.transpose((2, 0, 1))
    X.append(img)

    line = fd.readline()
    line = line[11:]
    line = line[0:-1]
    line = line.split(' ')
    line = filter(lambda l:l!='', line)
    line = list(line)
    line = list(map(int, line))
    y.append(line)

X = np.array(X)
y = np.array(y)
print(X.shape, y.shape)
np.savez('train.npz', X=X, y=y)
