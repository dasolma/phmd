import os
from phmd.readers import base as b
import pandas as pd


def read_file(f, file_path, _):
    X = pd.read_csv(f)
    X = X.ffill()
    X['rul'] = X.Timestamp.max() - X.Timestamp
    X['unit'] = os.path.split(file_path)[1].replace('.csv', '')
    del X['Timestamp']
    return X


def read_data(file_path, task: dict = None, filters: dict = None):
    def ffilter(dirs, files):
        return dirs, files

    X = b.read_dataset(file_path, None, ffilter=ffilter, fread_file=read_file, fprocess=None,
                          show_pbar=True, data_file_deep=2)

    X = X[0]

    return X
