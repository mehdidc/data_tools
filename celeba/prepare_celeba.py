from glob import glob
import os
import numpy as np
import h5py
from tqdm import tqdm
from scipy.misc import imread, imresize

np.random.seed(42)
filenames = glob(os.path.join("img_align_celeba_png", "*.png"))
np.random.shuffle(filenames)
w, h = 64, 64

def get_image(image_path):
    im = imread(image_path).astype(np.float)
    orig_h, orig_w = im.shape[:2]
    new_h = int(orig_h * w / orig_w)
    im = imresize(im, (new_h, w))
    margin = int(round((new_h - h)/2))
    return im[margin:margin+h]

with h5py.File('celeba64_align.h5', 'w') as f:
    dset = f.create_dataset("X", (len(filenames), 3, w, h), dtype='i8')
    for i, fname in tqdm(enumerate(filenames)):
        image = get_image(fname)
        image = image.transpose((2, 0, 1))
        dset[i] = image
