from skimage.io import imread_collection
from skimage.filters import threshold_otsu
from skimage.transform import resize
from skimage.util import pad
import os
import numpy as np
import h5py
from tqdm import tqdm

folder = "{}/iam/**/**/*.png".format(os.getenv("DATA_PATH"))
collection = imread_collection(folder)
collection = list(collection)
w_crop = 64
h_crop = 64
c, w, h = 1, 64, 64


def gen(nb):
    X_out = np.empty((nb, c, w, h))
    for i in range(nb):
        while True:
            img = np.random.choice(collection)
            ch = min(img.shape[0], h_crop)
            cw = min(img.shape[1], w_crop)
            crop_pos_y = np.random.randint(0, img.shape[0] - ch + 1)
            crop_pos_x = np.random.randint(0, img.shape[1] - cw + 1)
            x = crop_pos_x
            y = crop_pos_y
            im = img[y:y+ch, x:x+cw]
            im = im / 255.
            im = 1 - im
            #im = pad(im, 15, 'constant', constant_values=(0, 0))
            im = resize(im, (w, h))
            thresh = threshold_otsu(im)
            im = im > thresh
            im = im.astype(np.float32)
            if im.sum() > 500 and im.sum() < 2000:
                break
        X_out[i, 0] = im
    X_out = X_out.reshape((X_out.shape[0], -1))
    X_out = X_out.astype(np.float32)
    return X_out


batch_size = 100
nb_batches = 1000
total = batch_size * nb_batches
f = h5py.File('dataset.hdf5', 'w')
dataset = f.create_dataset('X', (total, w * h * c), maxshape=(None, w * h * c), compression="gzip")
k = 0
for i in tqdm(range(nb_batches)):
    dataset[k:k+batch_size] = gen(batch_size)
    k += batch_size
f.close()
