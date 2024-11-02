import scipy as scipy
from phmd.readers import base as b
import pandas as pd

FAULTS = ["healthy", "missing", "crack", "spall", "chip5a", "chip4a", "chip3a", "chip2a", "chip1a"]

def read_file(f, file_path, _):

    data = scipy.io.loadmat(f)
    data = data["AccTimeDomain"]

    Xs = []
    for unit in range(data.shape[1]):
        X = pd.DataFrame()
        X['acc'] = data[:, unit]
        X['fault'] = unit // 104
        X['unit'] = unit
        Xs.append(X)

    X = pd.concat(Xs)

    return X


def read_data(file_path, task: dict = None, filters: dict = None):

    def ffilter(dirs, files):
        return dirs, files

    X = b.read_dataset(file_path, None, ffilter=ffilter, fread_file=read_file, fprocess=None,
                       show_pbar=True, data_file_deep=3)

    return X[0]
