from phmd.readers import base as b
import pandas as pd
import numpy as np


def read_file(f, file_path, E=None, _=None):
    X = pd.read_csv(f)

    if 'errors' in file_path:
        X['errorID'] = X.errorID.map(lambda x: int(x.replace('error', '')) - 1)

    elif 'telemetry' in file_path:
        X = X.merge(E, on=['datetime', 'machineID'], how='left')
        X['unit'] = np.nan

        for machine in X.machineID.unique():
            x = X[X.machineID == machine]
            sample_cuts = [-1] + list(np.where(~x.failure.isnull())[0])

            for i, (l, r) in enumerate(zip(sample_cuts[:-1], sample_cuts[1:])):
                idx = x.index[l + 1:r+1]
                if idx.shape[0] > 0:
                    X.loc[idx, 'unit'] = X.iloc[idx].machineID.astype('str') + "_" + str(i + 1)
                    X.loc[idx, 'rul'] = np.arange(idx.shape[0])[::-1]
                    X.loc[idx, 'fault'] = X.iloc[idx].failure.values[-1]

        max_ruls = X.groupby('unit').rul.max()
        max_ruls = max_ruls[max_ruls > 100]
        X = X[X.unit.isin(max_ruls.index)]
        X['fault'] = X.fault.map(lambda x: int(x[-1]) - 1)

        del X['failure']

    return X


def read_data(file_path, task: dict = None, filters: dict = None):


    def ffilter(dirs, files):
        files = [f for f in files if 'failures' in f]

        return dirs, files

    E = b.read_dataset(file_path, None, ffilter=ffilter, fread_file=read_file, fprocess=None,
                          show_pbar=True, data_file_deep=3)
    E = E[0]

    def ffilter(dirs, files):
        files = [f for f in files if 'telemetry' in f]

        return dirs, files

    def read_telemetry(f, file_path, _):
        return read_file(f, file_path, E, _)

    X = b.read_dataset(file_path, None, ffilter=ffilter, fread_file=read_telemetry, fprocess=None,
                          show_pbar=True, data_file_deep=3)
    X = X[0]

    if task['target'] == 'fault':
        X = X[X.rul <= 100]

    return X
