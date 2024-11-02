import pandas as pd
import os
from phmd.readers import base as b

FILES = ['Genesis_normal', 'Genesis_lineardrive', 'Genesis_pressure']

def read_file(f, file_path, _):


    X = pd.read_csv(f)
    X['fault'] = FILES.index(os.path.split(file_path)[1].split('.')[0])
    X['unit'] = FILES.index(os.path.split(file_path)[1].split('.')[0])

    del X['Timestamp']

    return X



def read_data(file_path, task: dict = None, filters: dict = None):

    X = b.read_dataset(file_path, None, ffilter=None, fread_file=read_file, fprocess=None,
                          show_pbar=True, data_file_deep=2)

    # compute RUL


    return X
