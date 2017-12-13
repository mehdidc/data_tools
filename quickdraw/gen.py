import numpy as np
import glob
from clize import run
from sklearn.utils import shuffle

def main(*, nb_classes=None):
    X = []
    ys = []
    for filename in glob.glob('npy/*.npy'):
        classname = filename.split('/')[1].split('.')[0]
        try:
            data = np.load(filename)
        except Exception:
            continue
        X.append(data)
        ys.append([classname] * len(data))
    X = np.concatenate(X, axis=0)
    X = X.reshape((X.shape[0], 1, 28, 28))
    ys = np.concatenate(ys, axis=0)
    classnames = list(set(ys))
   
    if nb_classes is not None:
        nb_classes = int(nb_classes)
        classnb = [(ys==name).sum() for i, name in enumerate(classnames)]
        best = np.argsort(classnb)[::-1][0:nb_classes]
        cond = np.zeros(len(ys)).astype(bool)
        for b in best:
            print(classnames[b], classnb[b])
            cond = cond | (ys == classnames[b])
        ys = ys[cond]
        X = X[cond]
        classnames = [classnames[b] for b in best]

    class_to_int = {name: i for i, name in enumerate(classnames)}
    y = [class_to_int[name] for name in ys]
    y = np.array(y)
 
    X, y, ys = shuffle(X, y, ys, random_state=42)
    np.savez('data.npz', X=X, s=ys, y=y)

run(main)
