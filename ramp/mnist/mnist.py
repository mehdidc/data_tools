from subprocess import call
import numpy as np
import os

import pandas as pd
from skimage.io import imsave

def download(url):
    fname = os.path.basename(url)
    if not os.path.exists(fname):
        call('wget {}'.format(url), shell=True)

def convert(ids, X, out_img_folder='imgs'):
    for id_, x in zip(ids, X):
        imsave('{}/{}.png'.format(out_img_folder, id_), x)

def save_csv(ids, labels, out_csv):
    assert len(ids) == len(labels)
    cols = {
        'id' : ids,
        'class': labels,
    }
    pd.DataFrame(cols).to_csv(out_csv, index=False, columns=['id', 'class'])


def load_data():
    download('https://s3.amazonaws.com/img-datasets/mnist.npz')
    f = np.load('mnist.npz')
    X_train, Y_train = f['x_train'], f['y_train']
    X_test, Y_test = f['x_test'], f['y_test']
    return (X_train, Y_train), (X_test, Y_test)


if __name__ == '__main__':
    np.random.seed(42)
    (X_train, Y_train), (X_test, Y_test) = load_data()
    Y = np.concatenate((Y_train, Y_test), axis=0)
    ids = np.arange(0, len(X_train) + len(X_test))
    np.random.shuffle(ids)
    ids_train = ids[0:len(X_train)]
    ids_test = ids[len(X_train):]

    if not os.path.exists('imgs'):
        os.mkdir('imgs')
    
    convert(ids_train, X_train, out_img_folder='imgs')
    save_csv(ids_train, Y_train, out_csv='train.csv')

    convert(ids_test, X_test, out_img_folder='imgs')
    save_csv(ids_test, Y_test, out_csv='test.csv')

    save_csv(ids, Y, out_csv='full.csv')
