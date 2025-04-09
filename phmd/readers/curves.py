import os
import scipy as scipy
from phmd.readers import base as b
import pandas as pd
import pickle as pk
import random

def read_file(f, file_path, _):
    path = os.path.normpath(file_path)
    path_parts = path.split(os.sep)

    if '.pk' in file_path:
        data = pk.load(f)

        X = pd.DataFrame({
            'train_loss': data['loss'],
            'val_loss': data['val_loss'],
        })
        X['final_loss'] = data['val_loss'][-1]
        X['num_epochs'] = X.shape[0]

        run_hash = path_parts[-1].split('_')[-2]
        unit = f"{path_parts[-4]}_{path_parts[-3]}_{path_parts[-2]}_{run_hash[:5]}"
        X['unit'] = unit


    if '.csv' in file_path:
        X = pd.read_csv(f)
        X['unit'] = f"{path_parts[-4]}_{path_parts[-3]}_{path_parts[-2]}_" + X.run_hash.map(
            lambda x: x[:5])

    X['dataset'] = path_parts[-4]
    X['task'] = path_parts[-3]
    X['net'] = path_parts[-2]


    return X


def read_data(file_path, task: dict = None, filters: dict = {}):


    def ffilter(dirs, files):
        if filters is not None and 'data' in filters and filters['data'] == "curves":
            files = [f for f in files if '.pk' in f]
        if filters is not None and 'data' in filters and filters['data'] == "results":
            files = [f for f in files if '.csv' in f and 'processed' not in f]

        return dirs, files

    if (filters is not None) and (('data' in filters) and (filters['data'] == "curves") and
            os.path.exists(os.path.join(file_path, 'processed.csv'))):
        X = pd.read_csv(os.path.join(file_path, 'processed.csv'))
        X = X[~X.val_loss.isnull()]

    else:

        X = b.read_dataset(file_path, None, ffilter=ffilter, fread_file=read_file, fprocess=None,
                              show_pbar=True, data_file_deep=4)

        X = X[0]

        if (filters is not None) and 'data' in filters and filters['data'] == "curves":
            X['size'] = X.groupby(['unit']).val_loss.transform('size')
            X = X[X['size'] > 10]
            X = X[X.final_loss < 2]

            X['val_loss'] = X.val_loss.clip(-4, 4)

            del X['size']

        if (filters is not None) and ('data' in filters) and (filters['data'] == "curves"):
            X.to_csv(os.path.join(file_path, 'processed.csv'), index=False)


        if 'train_loss' in X.columns:
            X = X[~X.train_loss.isnull()]
            X = X[~X.val_loss.isnull()]

    return X
