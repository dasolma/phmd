import os
import scipy as scipy
from phmd.readers import base as b
import pandas as pd


FAULTS = ["norm", "spall", "crack"]

def read_file(f, file_path, _):
    letter, fault, number = os.path.split(file_path)[1].replace('.mat', '').split('_')
    unit = ord(letter) * 10000 + int(number) * 100 + FAULTS.index(fault)

    data = scipy.io.loadmat(f)

    vibration = data['TE_final'][:, 0]

    X = pd.DataFrame({'vibration': vibration})
    X['unit'] = unit
    X['fault'] = FAULTS.index(fault)

    return X


def read_data(file_path, task: dict = None, filters: dict = None):

    def ffilter(dirs, files):
        return dirs, files

    X = b.read_dataset(file_path, None, ffilter=ffilter, fread_file=read_file, fprocess=None,
                          show_pbar=True, data_file_deep=3)


    return X
