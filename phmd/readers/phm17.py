import pandas as pd
from phmd.readers import base as b

def _read_file(f, file_path, _, T=None):

    X = pd.read_csv(f)

    return X



def read_data(file_path, task: dict = None, filters: dict = None):


    def ffilter(dirs, files):
        files = [f for f in files if 'training.csv' in f]
        return dirs, files

    def read_file(f, file_path, _):
        return _read_file(f, file_path, _,)

    X = b.read_dataset(file_path, None, ffilter=ffilter, fread_file=read_file, fprocess=None,
                          show_pbar=True, data_file_deep=3)

    X = X[0]

    def ffilter(dirs, files):
        files = [f for f in files if 'training.csv' not in f]
        return dirs, files

    R = b.read_dataset(file_path, None, ffilter=ffilter, fread_file=read_file, fprocess=None,
                          show_pbar=True, data_file_deep=3)


    return X
