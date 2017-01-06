from clize import run
import numpy as np
import pandas as pd
from rdkit import Chem

def convert(filename, *, out='data.npz'):
    data = pd.read_hdf(filename, 'table')
    data = data.values
    np.savez(out, X=data)

if __name__ == '__main__':
    run(convert)
