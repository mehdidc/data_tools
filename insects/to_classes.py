import os
import pandas as pd
import glob
import subprocess
from clize import run


filename = 'test.csv'
out = 'test_img_classes'

df = pd.read_csv(filename)
df = df.set_index('id')

for i in range(18):
    folder = '{}/{}'.format(out, i)
    if not os.path.exists(folder):
        os.makedirs(folder)

for id_ in df.index.values:
    class_ = df.loc[id_]['class']
    filename = 'imgs/id_{}.jpg'.format(id_)
    dest = '{}/{}/{}.jpg'.format(out, class_, id_)
    if not os.path.exists(dest):
        os.link(filename, dest)
