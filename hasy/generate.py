import numpy as np
from skimage.io import imread
import glob
import pandas as pd
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from skimage.transform import resize
X = []
y = []
y_str = []
df = pd.read_csv('hasy-data-labels.csv')
df = df.set_index('path')
for filename in glob.glob('hasy-data/*.png'):
    x = imread(filename)
    x = resize(x, (28, 28), preserve_range=True)
    x = x[np.newaxis, np.newaxis, :, :, 0]
    x = 255 - x
    X.append(x)
    y.append(df.loc[filename]['symbol_id'])
    y_str.append(df.loc[filename]['latex'])
X = np.concatenate(X, axis=0)
y_str = np.array(y_str)
y = LabelEncoder().fit_transform(y_str)
print(X.shape, y.shape, y_str.shape)
X, y, y_str = shuffle(X, y, y_str, random_state=42)
np.savez('data.npz', X=X, y=y, y_str=y_str)
