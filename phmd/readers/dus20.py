import os
from phmd.readers import base as b
import pandas as pd


COL_NAMES = ["ID", "Date_Time", "POSIX", "Sensor ID", "Measured variable", "acc"]

def read_file(f, file_path, _):
    unbalance = int(os.path.split(file_path)[1].split("_")[2])
    unit = '_'.join(os.path.split(file_path)[1].split("_")[:2])
    X = pd.read_csv(f, names=COL_NAMES)
    X['unbalance'] = unbalance
    X['unit'] = unit
    

    return X


def read_data(file_path, task: dict = None, filters: dict = None):

    def ffilter(dirs, files):
        return dirs, files

    X = b.read_dataset(file_path, None, ffilter=ffilter, fread_file=read_file, fprocess=None,
                          show_pbar=True, data_file_deep=2)

    X = X[0]

    units = sorted(X.unit.unique())
    X['unit'] = X.unit.map(lambda x: units.index(x)).astype('int8')

    return X
