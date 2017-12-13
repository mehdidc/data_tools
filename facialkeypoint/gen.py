import numpy as np
import pandas as pd

df = pd.read_csv('test.csv')
img = df['Image']
img = img.values
for im in img:
    im = im.split(' ')
    print(len(im))
    break
