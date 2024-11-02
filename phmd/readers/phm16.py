import pandas as pd
from phmd.readers import base as b
import numpy as np

def read_file(f, file_path, bearing):

    X = pd.read_csv(f)

    if 'MACHINE_ID' in X.columns:
        del X['MACHINE_ID']

    return X



def read_data(file_path, task: dict = None, filters: dict = None):


    def ffilter(dirs, files):
        files = [f for f in files if 'removalrate' not in f]
        return dirs, files

    X = b.read_dataset(file_path, None, ffilter=ffilter, fread_file=read_file, fprocess=None,
                          show_pbar=True, data_file_deep=3)

    X = X[0]

    def ffilter(dirs, files):
        files = [f for f in files if 'removalrate' in f]
        return dirs, files

    R = b.read_dataset(file_path, None, ffilter=ffilter, fread_file=read_file, fprocess=None,
                          show_pbar=True, data_file_deep=3)

    WAFER_MAXTIME = X.groupby(['WAFER_ID', 'STAGE']).TIMESTAMP.max().reset_index()
    R = pd.merge(R[0], WAFER_MAXTIME)

    X = pd.merge(X, R)
    X['WAFER_ID'] = np.arange(X.shape[0])
    X['STAGE'] = X.STAGE.map(lambda x: 0 if x == 'A' else 1).astype('int64')
    X['MACHINE_DATA'] = X.MACHINE_DATA.astype('int64')
    return X
