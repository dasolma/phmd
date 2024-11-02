import random
import h5py
from phmd.readers import base as b
import pandas as pd


TYPES = {'K001': ('Healthy', 'Normal'), 'K002': ('Healthy', 'Normal'), 'K003': ('Healthy', 'Normal'),
         'K004': ('Healthy', 'Normal'), 'K005': ('Healthy', 'Normal'), 'K006': ('Healthy', 'Normal'),
         'KA01': ('Artificial', 'OR'), 'KA03': ('Artificial', 'OR'), 'KA05': ('Artificial', 'OR'),
         'KA06': ('Artificial', 'OR'), 'KA07': ('Artificial', 'OR'), 'KA09': ('Artificial', 'OR'),
         'KI01': ('Artificial', 'IR'), 'KI03': ('Artificial', 'IR'), 'KI05': ('Artificial', 'IR'),
         'KI07': ('Artificial', 'IR'), 'KI08': ('Artificial', 'IR'), 'KA04': ('Real', 'OR'),
         'KA15': ('Real', 'OR'), 'KA16': ('Real', 'OR'), 'KA22': ('Real', 'OR'), 'KA30': ('Real', 'OR'),
         'KB23': ('Real', 'IR'), 'KB24': ('Real', 'IR'), 'KI04': ('Real', 'IR'), 'KI14': ('Real', 'IR'),
         'KI16': ('Real', 'IR'), 'KI17': ('Real', 'IR'), 'KI18': ('Real', 'IR'), 'KI21': ('Real', 'IR'),
         'KB27': ('Real', 'OR')}

COMPONENT = ['Normal', 'IR', 'OR']
DTYPE = ['Healthy', 'Artificial', 'Real']

def read_file(f, file_path, bearing):

    D = h5py.File(f, 'r')

    X = pd.DataFrame()
    for c in D.keys():
        if c == 'unit':
            X[c] = [v.decode() for v in D[c][:]]
        else:
            X[c] = D[c][:]

    return X


def read_data(file_path: str, task: dict = None, filters: dict = None):


    def ffilter(dirs, files):
        return dirs, files

    X = b.read_dataset(file_path, None, ffilter=ffilter, fread_file=read_file, fprocess=None,
                       show_pbar=True, data_file_deep=3, sort_files=False)

    X = X[0]

    # split healthy units
    healty_units = list(X[X['type'] == DTYPE.index('Healthy')].unit.unique())
    healty_test_units = random.sample(healty_units, int(len(healty_units) * 0.4))
    healty_train_units = set(healty_units) - set(healty_test_units)
    if task['target'] == 'fault_component_1':
        X.rename({'fault_component': 'fault_component_1'}, axis='columns', inplace=True)

        # create train and test sets
        X_test = X[X['unit'].isin(healty_test_units) | (X['type'] == DTYPE.index('Real')) ]
        X_train = pd.concat((X[X['unit'].isin(healty_train_units)], X[X['type'] == DTYPE.index('Artificial')]))
        return (X_train, X_test)

    if task['target'] == 'fault_component_2':
        X.rename({'fault_component': 'fault_component_2'}, axis='columns', inplace=True)

        # split real units
        real_units = list(X[X['type'] == DTYPE.index('Real')].unit.unique())
        real_test_units = random.sample(real_units, int(len(real_units) * 0.4))
        real_train_units = set(real_units) - set(real_test_units)

        # create train and test sets
        X_test = pd.concat((X[X['unit'].isin(healty_test_units)], X[X['unit'].isin(real_test_units)]))
        X_train = pd.concat((X[X['unit'].isin(healty_train_units)],X[X['unit'].isin(real_train_units)]))
        return (X_train, X_test)
    
    return X
