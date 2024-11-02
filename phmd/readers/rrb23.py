from phmd.readers import base as b
import pandas as pd
import os
import numpy as np


def read_file(f, file_path, _):
    unit = os.path.split(file_path)[1].replace('.csv', '')
    X = pd.read_csv(f)
    X['mode'] = X['mode'].clip(-1, 0)

    aux = np.vstack((X['mode'].values[:-1], X['mode'].values[1:])).T
    aux = aux[:, 0] != aux[:, 1]
    aux = np.append(aux, [False])
    aux = np.cumsum(aux)
    
    X['cycle'] = aux
    X['rul'] = X.cycle.max() - X.cycle
    X['unit'] = unit
    
    X = X[(X['mode'] == -1) & (X.mission_type ==1)].reindex()
    X = X[X.index % 10 == 0].copy()
    
    
    return X


def read_data(file_path, task: dict = None, filters: dict = None):

    def ffilter(dirs, files):
        return dirs, files

    X = b.read_dataset(file_path, None, ffilter=ffilter, fread_file=read_file, fprocess=None,
                          show_pbar=True, data_file_deep=3)


    return X
