import os
from phmd.readers import base as b
import pandas as pd


FAULTS = ["normal", "bearing", "looseness", "unbalance"]

def read_file(f, file_path, _):
    fault = os.path.split(file_path)[1].split("_")[-1].replace('.csv', '').lower()
    unit = int(os.path.split(file_path)[1].split("_")[1][0]) + (FAULTS.index(fault) / 10)

    X = pd.read_csv(f)
    if len(X.columns) == 3:
        del X[X.columns[0]]

    step_size = 5
    X = X.iloc[::step_size, :]
    
    X.columns = ["c1", "c2"]

    X['unit'] = unit
    X['fault'] = FAULTS.index(fault)

    return X


def read_data(file_path, task: dict = None, filters: dict = None):

    def ffilter(dirs, files):
        files =[f for f in files if 'high' not in f]
        return dirs, files

    X = b.read_dataset(file_path, None, ffilter=ffilter, fread_file=read_file, fprocess=None,
                          show_pbar=True, data_file_deep=3)

    X = X[0]
    return X
