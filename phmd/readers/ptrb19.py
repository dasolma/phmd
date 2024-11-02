import os
import scipy as scipy
from phmd.readers import base as b
import pandas as pd

FAULTS = ["NO", "IR", "RO"]

HEADERS = ["acc1_x", "acc1_y", "acc1_z", "acc2_x", "acc2_y", "acc2_z"]

def read_file(f, file_path, _, Xp=None):
    key = os.path.split(file_path)[1].replace('.mat', '')

    data = scipy.io.loadmat(f)

    X = pd.DataFrame()
    for i, col in enumerate(HEADERS):
        X[col] = data[key][:, i]

    key = key.split("_")
    n = int(key[0][1])
    X['unit'] = float(key[0][1] + "." + ''.join(key[1:]))
    if n == 0:
        X['fault'] = 0
    elif n <= 3:
        X['fault'] = 1
    else:
        X['fault'] = 2

    return X


def read_data(file_path, task: dict = None, filters: dict = None):

    def ffilter(dirs, files):
        files = [f for f in files if 'FileNames' not in f]

        return dirs, files

    X = b.read_dataset(file_path, None, ffilter=ffilter, fread_file=read_file, fprocess=None,
                       show_pbar=True, data_file_deep=3, concat_in_file=True)

    return X[0]
