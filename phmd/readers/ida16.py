import pandas as pd
from phmd.readers import base as b
import numpy as np



def read_file(f, file_path, bearing):

    data = f.read()
    if isinstance(data, bytes):
        data = data.decode()

    data = data.split('\n')[20:]

    cols = data[0].strip().split(',')
    data = [l.strip().split(',') for l in data[1:] if len(l) > 0]

    X = pd.DataFrame(data, columns=cols)

    X['fault'] = X['class'].map(lambda x: 0 if x == 'neg' else 1)
    X.replace('na', np.nan, inplace=True)
    X['unit'] = np.arange(X.shape[0])

    for c in X.columns[1:-1]:
        X[c] = X[c].astype(np.float32)

    X.fillna(-1, inplace=True)
    

    return X



def read_data(file_path, task: dict = None, filters: dict = None):

    def ffilter(dirs, files):
        return dirs, files

    X = b.read_dataset(file_path, None, ffilter=ffilter, fread_file=read_file, fprocess=None,
                          show_pbar=True, data_file_deep=4)

    # compute RUL


    return X
