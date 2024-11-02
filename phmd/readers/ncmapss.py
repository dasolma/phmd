from phmd.readers import base as b
import pandas as pd
import numpy as np
import h5py
import os

from phmd.readers.base import ensure_iterable


def read_file(fp, file_path, data, type='train', filters=None):
    suffix = 'dev' if type == 'train' else 'test'
    # Load data
    unit1 = int(os.path.basename(file_path).split('.')[0].split("_")[1].split('-')[0].replace('DS', '').
                replace('a', '').replace('b', '').replace('c', ''))
    parts = os.path.basename(file_path).split('.')[0].split("_")[1].split('-')
    unit2 = 0
    if len(parts) == 2:
        unit2 = int(os.path.basename(file_path).split('.')[0].split("_")[1].split('-')[1])
    FD = int(f"{unit1}{unit2:03d}")

    with h5py.File(fp, 'r') as hdf:
        # features set
        W = np.array(hdf.get('W_%s' % suffix))  # W
        X_s = np.array(hdf.get('X_s_%s' % suffix))  # X_s
        A = np.array(hdf.get('A_%s' % suffix))  # Auxiliary

        # target
        Y = np.array(hdf.get('Y_%s' % suffix))  # RUL

        # Varnams
        W_var = np.array(hdf.get('W_var'))
        X_s_var = np.array(hdf.get('X_s_var'))
        A_var = np.array(hdf.get('A_var'))

        # from np.array to list dtype U4/U5
        W_var = list(np.array(W_var, dtype='U20'))
        X_s_var = list(np.array(X_s_var, dtype='U20'))
        A_var = list(np.array(A_var, dtype='U20'))

    #FD = np.array([FD] * Y.shape[0], dtype='float16').reshape(Y.shape)

    X = np.concatenate((W, X_s, A, Y), axis=1)
    X = pd.DataFrame(data=X, columns=W_var + X_s_var + A_var + ['rul'])
    X['unit_bck'] = X.unit
    X['unit'] = FD + (X.unit / 100)



    if filters is not None:
        if 'Fc' in filters:
            Fc = ensure_iterable(filters['Fc'])
            X = X[X.Fc.isin(Fc)]
    
    for c in ['alt', 'Mach', 'TRA', 'T2', 'T24', 'T30', 'T48', 'T50', 'P15', 'P2',
       'P21', 'P24', 'Ps30', 'P40', 'P50', 'Nf', 'Nc', 'Wf', 'Fc', 'hs']:
        X[c] = X[c].astype('float16')

    return X


def read_train(file_path, task: dict = None, filters: dict = None, ruls=None):

    reader = lambda filename, file_path, data, Xp=None: read_file(filename, file_path, data, 'train', filters)

    X = b.read_dataset(file_path, task, fread_file=reader, fprocess=None, concat_delegated=True,
                       concat_in_file=True)
    return X


def read_test(file_path, task: dict = None, filters: dict = None, ruls=None):
    reader = lambda filename, file_path, data: read_file(filename, file_path, data, 'test', filters)

    X = b.read_dataset(file_path, task, fread_file=reader, fprocess=None)
    return X
