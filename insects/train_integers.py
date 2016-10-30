from sklearn.cross_validation import StratifiedShuffleSplit
import numpy as np
def read_data(npz_filename):
    data = np.load(npz_filename)
    X_array = data['X']
    y_array = data['y']
    return X_array, y_array

def prepare_data():
    raw_filename = "/home/mcherti/ramp_pollenating_insects/databoard_pollenating_insects_2170/data/raw/data_64x64.npz"
    X_array, y_array = read_data(raw_filename)

    cv = StratifiedShuffleSplit(
        y_array, 1, test_size=0.2, random_state=57)
    train_is, test_is = list(cv)[0]

    for a in train_is:
        print(a)
prepare_data()
