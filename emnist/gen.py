import numpy as np
# Import the relevant modules to be used later
import gzip
import os
import struct
# Config matplotlib for inline plotting

def loadData(gzfname, cimg):
    with gzip.open(gzfname) as gz:
        n = struct.unpack('I', gz.read(4))
        # Read magic number.
        if n[0] != 0x3080000:
            raise Exception('Invalid file: unexpected magic number.')
        # Read number of entries.
        n = struct.unpack('>I', gz.read(4))[0]
        if n != cimg:
            raise Exception('Invalid file: expected {0} entries.'.format(cimg))
        crow = struct.unpack('>I', gz.read(4))[0]
        ccol = struct.unpack('>I', gz.read(4))[0]
        print(crow, ccol)
        if crow != 28 or ccol != 28:
            raise Exception('Invalid file: expected 28 rows/cols per image.')
        # Read data.
        res = np.fromstring(gz.read(cimg * crow * ccol), dtype = np.uint8)
    res = res.reshape((cimg, 1, crow, ccol))
    res = res.transpose((0, 1, 3, 2))
    return res

def loadLabels(gzfname, cimg):
    with gzip.open(gzfname) as gz:
        n = struct.unpack('I', gz.read(4))
        # Read magic number.
        if n[0] != 0x1080000:
            raise Exception('Invalid file: unexpected magic number.')
        # Read number of entries.
        n = struct.unpack('>I', gz.read(4))
        if n[0] != cimg:
            raise Exception('Invalid file: expected {0} rows.'.format(cimg))
        # Read labels.
        res = np.fromstring(gz.read(cimg), dtype = np.uint8)
    return res.reshape((cimg,))

if __name__ == '__main__':
    X_train = loadData('gzip/emnist-balanced-train-images-idx3-ubyte.gz', 112800)
    y_train = loadLabels('gzip/emnist-balanced-train-labels-idx1-ubyte.gz', 112800)

    X_test = loadData('gzip/emnist-balanced-test-images-idx3-ubyte.gz', 18800)
    y_test = loadLabels('gzip/emnist-balanced-test-labels-idx1-ubyte.gz', 18800)

    print(len(np.unique(y_train)))
    np.savez('data_train.npz', X=X_train, y=y_train)
    np.savez('data_test.npz', X=X_test, y=y_test)
