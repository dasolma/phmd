import os
import scipy as scipy
from phmd.readers import base as b
import pandas as pd




def read_file(f, file_path, _):

    fault = os.path.split(os.path.split(file_path)[0])[1] == 'fault'

    unit = int(os.path.split(file_path)[1].replace('.mat', '').split('_')[1]) + (100 if fault else 0)

    data = scipy.io.loadmat(f)

    vibration = data['gs'][:, 0]

    X = pd.DataFrame({'vibration': vibration})
    X['unit'] = unit
    X['fault'] = int(fault)

    return X


def read_data(file_path, task: dict = None, filters: dict = None):

    def ffilter(dirs, files):
        return dirs, files

    X = b.read_dataset(file_path, None, ffilter=ffilter, fread_file=read_file, fprocess=None,
                          show_pbar=True, data_file_deep=3)


    return X
