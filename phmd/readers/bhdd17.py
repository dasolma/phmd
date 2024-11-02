import pandas as pd
from phmd.readers import base as b
import h5py
import numpy as np

def read_file(f, file_path, bearing):

    f = h5py.File(f, 'r')
    
    cols = [i.decode() for i in f['data']['block0_items']] + [i.decode() for i in f['data']['block1_items']]
    X = np.hstack((f['data']['block0_values'][:], f['data']['block1_values'][:]))
    
    X = pd.DataFrame(X, columns=cols)
    X.rename({'serial-number': 'unit'}, axis='columns', inplace=True)
    X.rename({'Drive Status': 'fault'}, axis='columns', inplace=True)
    X['fault'] = 1 - X.fault.clip(0, 1)

    return X



def read_data(file_path, task: dict = None, filters: dict = None):

    def ffilter(dirs, files):
        files = list(filter(lambda f: '.h5' in f, files))
        return dirs, files

    X = b.read_dataset(file_path, None, ffilter=ffilter, fread_file=read_file, fprocess=None,
                          show_pbar=True, data_file_deep=3)



    return X
