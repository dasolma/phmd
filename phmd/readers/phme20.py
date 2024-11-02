import os
from phmd.readers import base as b
import pandas as pd


FAULTS = ["norm", "spall", "crack"]

def _read_file(f, file_path, _, O=None):
    X = pd.read_csv(f)

    if O is not None:
        sample = float(os.path.split(file_path)[1].split('.')[0].replace('Sample', ''))
        X['Sample'] = sample
        X = X.merge(O, on="Sample")
        X['rul'] = X['Time(s)'].max() - X['Time(s)']
        X['unit'] = X.Sample
        del X['Sample']


    return X


def read_data(file_path, task: dict = None, filters: dict = None):

    def ffilter(dirs, files):
        files = [f for f in files if 'operation_profiles' in f]
        return dirs, files

    O = b.read_dataset(file_path, None, ffilter=ffilter, fread_file=_read_file, fprocess=None,
                          show_pbar=True, data_file_deep=3)[0]
    def ffilter(dirs, files):
        files = [f for f in files if 'operation_profiles' not in f]
        return dirs, files

    def read_file(f, file_path, _):
        return _read_file(f, file_path, _, O)

        return X

    X = b.read_dataset(file_path, None, ffilter=ffilter, fread_file=read_file, fprocess=None,
                          show_pbar=True, data_file_deep=3)

    X = X[0]
    return X
