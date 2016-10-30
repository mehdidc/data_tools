import click
import pandas as pd
import json
from functools import partial
import numpy as np
from collections import Counter
from skimage.io import imsave
from skimage import draw
import cv2
from sklearn.preprocessing import LabelEncoder
from skimage.exposure import is_low_contrast
from skimage.transform import resize

def parse_line(line):
    tokens = line.split(';')
    assert len(tokens)>=3
    symbol_id, user_id, data = tokens[0:3]
    return {'class': int(symbol_id), 'data': data, 'user_id': int(user_id), 'symbol_id': int(symbol_id)}

def points_from_line(line_data, w=32, h=32):
    data =  line_data['data']
    class_ = line_data['class']
    data = json.loads(data)
    return data

def to_image(points, w=32, h=32, x_max=None, y_max=None, tickness=1, pad=8):
    all_points = [p for phase in points for p in phase]
    x_max = max([p['x'] for p in all_points]) + pad
    y_max = max([p['y'] for p in all_points]) + pad
    x_min = min([p['x'] for p in all_points]) - pad
    y_min = min([p['y'] for p in all_points]) - pad
    P = np.zeros((h, w), dtype=np.uint8)
    for phase in points:
        xprev = None
        yprev = None
        for p in phase:
            x, y = p['x'], p['y']
            x = float(x)
            y = float(y)
            x = (x - x_min) / (x_max - x_min)
            y = (y - y_min) / (y_max - y_min)
            x = x * w
            x = int(x)
            y = y * h
            y = int(y)
            x = min(x, w - 1)
            y = min(y, h - 1)
            if xprev and yprev:
                #rr, cc, val = draw.line_aa(yprev, xprev, y, x)
                #rr, cc = draw.bezier_curve(yprev, xprev, (yprev+y)/2, (xprev+x)/2, y, x, 1)
                cv2.line(P, (xprev, yprev), (x, y), 255, tickness, cv2.LINE_AA)
            xprev = x
            yprev = y
    return P

@click.command()
@click.option('--w', default=32, required=False)
@click.option('--h', default=32, required=False)
@click.option('--pad', default=3, required=False)
@click.option('--impad', default=0, required=False)
@click.option('--tickness', default=1, required=False)
@click.option('--most_common_classes', default=0, required=False)
@click.option('--out', default='data.npz', required=False)
def rasterize(w, h, pad, impad, tickness, most_common_classes, out):
    mapping = get_symbol_mapping()
    
    fd = open('train-data.csv')
    lines = fd.readlines()[1:]
    
    fd = open('test-data.csv')
    lines += fd.readlines()[1:]

    lines = map(parse_line, lines)
    lines = filter(lambda l:l['user_id']==16925, lines)

    y_str = [mapping[l['symbol_id']] for l in lines]
    y_str = np.array(y_str)
    y = [l['symbol_id'] for l in lines]
    y = np.array(y)
        
    points = map(points_from_line, lines)
    X = map(partial(to_image, w=w, h=h, tickness=tickness, pad=pad), points)
    for i in range(len(X)):
        if impad>0:
            X[i] = np.pad(X[i], impad, 'constant')
            X[i] = resize(X[i], (h, w), preserve_range=True)

    mask = np.ones(len(y), dtype=np.bool)
    if most_common_classes:
        counter = Counter(y)
        all_classes = set(y)
        common_classes = set(y_id for y_id, _ in counter.most_common(most_common_classes))
        filtered_classes = set(all_classes) - set(common_classes)
        for y_id in filtered_classes:
            mask[y==y_id] = False
    low_constrast_mask = map(lambda i:is_low_contrast(X[i]), np.arange(len(X)))
    low_constrast_mask = np.array(low_constrast_mask, dtype=np.bool) 
    mask = mask & (~low_constrast_mask)

    X = np.array(X)
    X = X[mask]
    y = y[mask]
    y_str = y_str[mask]
    y = LabelEncoder().fit_transform(y)
    i = 0
    X = X[:, None, :, :]
    np.savez(out, X=X, y=y, y_str=y_str)

def get_symbol_mapping():
    df = pd.read_csv('symbols.csv', sep=';')
    mapping = {sid: latex for sid, latex in zip(df['symbol_id'], df['latex'])}
    return mapping

if __name__ == '__main__':
    rasterize()
