import pandas as pd
import os
import tqdm
from phmd.readers import base as b
import numpy as np


def repeat_values(data, size):
    repeats = int(6000 / size)
    if repeats > 1:
        data = [r for v in data for r in [v] * repeats]
    return data

def read_file(f, file_path, bearing, task):

    feature = os.path.split(file_path)[1][:-4]

    X = pd.read_csv(f, sep='\t')

    #cycles = [j for i in range(2204) for j in [i] * 6000]

    if feature == 'profile':
        feature = ['cooler', 'valve', 'pump', 'accumulator']
        data = {}
        if task is None:
            for i, target in enumerate(feature):
                data[target] = repeat_values(X.values[:, i], 1)
        else:
            i = feature.index(task['target'])
            labels = list(np.unique(X.values[:, i]))
            encoded = [labels.index(l) for l in X.values[:, i]]
            data[task['target']] = repeat_values(encoded, 1)

            feature = task['target']

        X = pd.DataFrame(data)

    else:
        size = X.shape[1]

        data = X.values.flatten()

        data = repeat_values(data, size)

        X = pd.DataFrame({feature: data})

    return X.T, feature


def read_data(file_path, task: dict = None, filters: dict = None):

    def ffilter(dirs, files):
        return dirs, files

    cols = []
    def task_reader(f, file_path, bearing):
        X, col = read_file(f, file_path, bearing, task)

        if isinstance(col, list):
            for c in col:
                cols.append(c)
        else:
            cols.append(col)

        return X

    X = b.read_dataset(file_path, None, ffilter=ffilter,
                       fread_file=task_reader, fprocess=None,
                          show_pbar=True, data_file_deep=3)

    X = X[0]
    X = X.T
    X.columns = [c.lower() for c in cols]

    X['cycle'] = [j for i in range(2204) for j in [i] * 6000]

    return X
