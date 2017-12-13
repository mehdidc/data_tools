import os
import pandas as pd
import glob
import subprocess
from clize import run


filename = 'train.csv'
out = 'train_img_classes_with_folder'
with_folder = True

df = pd.read_csv(filename)
df = df.set_index('id')

for i in range(18):
    if with_folder:
        folder = '{}/{}/imgs'.format(out, i)
    else:
        folder = '{}/{}'.format(out, i)
    if not os.path.exists(folder):
        os.makedirs(folder)

for id_ in df.index.values:
    class_ = df.loc[id_]['class']
    filename = 'imgs/id_{}.jpg'.format(id_)
    if with_folder:
        dest = '{}/{}/imgs/{}.jpg'.format(out, class_, id_)
    else:
        dest = '{}/{}/{}.jpg'.format(out, class_, id_)
    if not os.path.exists(dest):
        os.link(filename, dest)
