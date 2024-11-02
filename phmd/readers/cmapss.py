import os
from phmd.readers import base as b
import pandas as pd

SETTING_COLS = ["Altitude", "TRA", "Mach_number"]
SENSOR_COLS = ["T2", "T24", "T30", "T50", "P2", "P15", "P30", "Nf", "Nc", "epr", "Ps30", "phi", 
               "NRf", "NRc", "BPR", "farB", "htBleed", "Nf_dmd", "PCNfR_dmd", "W31", "W32"]

HEADERS = ['id', 'cycle'] + SETTING_COLS + SENSOR_COLS

FAULT_TYPES = {
    "FD001": 1,
    "FD002": 1,
    "FD003": 2,
    "FD004": 2,
}

OPERATING_CONDITIONS = {
    "FD001": 1,
    "FD002": 6,
    "FD003": 1,
    "FD004": 6,
}

def read_file(f, file_path, data, ruls=None):

    file_name = os.path.split(file_path)[1]
    fd = int(file_name.split('_')[1].split('.')[0].replace('FD', ''))
    fdfull = file_name.split('_')[1].split('.')[0]
    X = pd.read_csv(f, sep=" ", header=None)
    if len(X.columns) == 28:
        X.drop([26, 27], axis = 'columns', inplace=True)
    X.columns = HEADERS
    X['FD'] = fd
    X['Condition'] = OPERATING_CONDITIONS[fdfull]
    X['Fault_type'] = FAULT_TYPES[fdfull]

    if "train" in file_path: # compute RUL
        X['max_cycle'] = X.groupby('id').cycle.transform('max')
        X['rul'] = X.max_cycle - X.cycle
        del X['max_cycle']

    if ruls is not None:
        X = pd.merge(X, ruls, on=['FD', 'id'], how='left')
        assert ~X.isnull().any().any()

        X['rul'] = X.max_cycle - X.cycle
        del X['max_cycle']

    return X

def read_rul(f, file_path, data):
    fd = int(file_path.split('_')[-1].split('.')[0].replace('FD', ''))
    ruls = [int(rul) for rul in f.readlines()]

    return pd.DataFrame(data={'max_cycle': ruls, 'id': list(range(1, len(ruls) + 1)), 'FD': [fd] * len(ruls)})


def read_data(file_path, task: dict = None, filters: dict = None):

    ruls = None
    if 'test' in file_path:
        def filter_rul_files(dirs, files):
            files = [f for f in files if 'RUL' in f]
            return dirs, files

        ruls = b.read_dataset(file_path, None, ffilter=filter_rul_files, fread_file=read_rul, fprocess=None,
                              show_pbar=False)[0]

    def ffilter(dirs, files):
        files = [f for f in files if 'RUL' not in f]
        return dirs, files

    def __read_file(f, file_path, data):
        return read_file(f, file_path, data, ruls=ruls)

    X = b.read_dataset(file_path, None, ffilter=ffilter, fread_file=__read_file, fprocess=None,
                          show_pbar=True, data_file_deep=3)


    return X

def process(Xs, ruls=None):
    if ruls is None:
        ruls = [None for _ in range(len(Xs))]

    for i in range(len(Xs)):
        X, rul = Xs[i], ruls[i]

        ttol = X.groupby('id').cycle.max().reset_index()
        ttol.columns = ['id', 'ttol']

        # compute RUL
        X = X.merge(ttol, on='id')
        X['rul'] = X.ttol - X.cycle

        if rul is not None:
             X = X.merge(rul, on=['FD', 'id'], how='left')
             X['rul'] = X.rul + X.rrul
             X.drop(['rrul'], axis = 1, inplace=True)

        X.drop(['ttol'], axis = 1, inplace=True)
        X['rul'] = X.rul.astype('int32')

        Xs[i] = X

    return Xs


def __read_data(file_path, task:dict = None, filters:dict = None, ruls=None):
    ffilter = lambda d, f: filter_files(d, f, filters)
    fprocess = lambda X: process(X, ruls=ruls)

    X =  b.read_dataset(file_path, task, ffilter= ffilter, fread_file=read_file, fprocess=fprocess)
    return X

read_train = __read_data

def read_test(file_path, task:dict = None, filters:dict = None):
    def read_rul(f, file_path, data):
        fd = int(file_path.split('_')[1].split('.')[0].replace('FD', ''))
        ruls = [int(rul) for rul in f.readlines()]

        return pd.DataFrame(data={'rrul': ruls, 'id': list(range(1, len(ruls)+1)), 'FD': [fd] * len(ruls)})

    def filter_rul_files(dirs, files):
        isRULData = lambda path: path.split("_")[0][-3:] == 'RUL'
        data_files = sorted(f for f in files if isRULData(f))
        dirs = set('/'.join(f.split('/')[:-1]) for f in data_files)

        return dirs, data_files

    ruls = b.read_dataset(file_path, None, ffilter=filter_rul_files, fread_file=read_rul, fprocess=None, show_pbar=False)

    X = __read_data(file_path, task, filters, ruls)

    return X