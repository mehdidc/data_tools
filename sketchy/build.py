import glob
import os
from skimage.io import imread, imsave
from skimage.transform import resize
import numpy as np
import pandas as pd
x = []
y = []
filenames = glob.glob('256x256/sketch/tx_000000000000/**/*.png')
print(len(filenames))
for f in filenames: 
    category = f.split('/')[-2]
    img = imread(f)
    img = resize(img, (64, 64))
    img = img[:, :, 0]
    img = img[None, None, :, :]
    x.append(img)
    y.append(category)
y_str = np.array(y)
x = np.concatenate(x, axis=0)
y_uniq = list(set(y))
y_int_uniq = range(len(y_uniq))
y_str_to_int = {s:i for s, i in zip(y_uniq, y_int_uniq)}
pd.DataFrame({'symbol': y_uniq, 'id': y_int_uniq}).to_csv('symbols.csv')
y_int = [y_str_to_int[ys] for ys in y]
y_int = np.array(y_int)
np.savez('data.npz', X=x, y=y_int, y_str=y_str)
