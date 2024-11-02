import pandas as pd
import os
import scipy as scipy
from phmd.readers import base as b
from collections import defaultdict

def read_file(f, file_path, _):
    unit = os.path.split(file_path)[1].replace('.mat', '')

    data = scipy.io.loadmat(f)

    columns = [t[0] for t in eval(str(data['data'][0, 0][5][0].dtype))]

    data_dict = defaultdict(lambda: [])
    
    L = data['data'][0, 0][5][0].shape[0]
    for i in range(L):
        n = max([d.shape[1] if len(d.shape) > 1 else d.shape[0] for d in data['data'][0, 0][5][0, i]])
        for j, c in enumerate(columns):
            d = data['data'][0, 0][5][0, i][j].flatten()
            if len(d) != n:
                data_dict[c] += [d[0]] * n
            else:
                data_dict[c] += list(d)

    X = pd.DataFrame(data_dict)
    X['rul'] = X.cycle.values.max() - X.cycle.values
    X = X[['voltage', 'current', 'temperature', 'capacity', 'energy', 'rul']]
    X['unit'] = unit.lower()

    return X


def read_data(file_path, task: dict = None, filters: dict = None):

    def ffilter(dirs, files):
        return dirs, files

    X = b.read_dataset(file_path, None, ffilter=ffilter, fread_file=read_file, fprocess=None,
                          show_pbar=True, data_file_deep=3)


    return X
