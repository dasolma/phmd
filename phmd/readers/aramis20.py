import scipy as scipy
from phmd.readers import base as b
import pandas as pd
import numpy as np
from collections import defaultdict

def read_file(f, file_path, _):
    mat_data = scipy.io.loadmat(f)

    data = defaultdict(lambda: np.array([]))
    for m in range(200):

        for j in range(4):

            for i in range(10):
                signal = mat_data['Train_data'][0][m][0][0, j][i]
                data[f's{i}'] = np.concatenate((data[f's{i}'], signal))

            where_fail = np.where(mat_data['Train_data'][0][m][1][j])[0]
            if len(where_fail) > 0:
                tlife = np.where(mat_data['Train_data'][0][m][1][j])[0].min()
                rul = np.clip(tlife - np.arange(0,  len(signal)), 0, tlife)
            else:
                rul = np.zeros((len(signal), ))

            data["fault"] = np.concatenate((data["fault"],
                                            mat_data['Train_data'][0][m][1][j]))
            data["component"] = np.concatenate((data["component"],
                                                np.repeat([m+(j/10)], len(signal))))
            data["unit"] = np.concatenate((data["unit"],
                                                np.repeat([m], len(signal))))

            data["rul"] = np.concatenate((data["rul"], rul))

            assert len(data["fault"]) == len(data["component"]) == len(data["unit"])

    X = pd.DataFrame(data)
    return X


def read_data(file_path, task: dict = None, filters: dict = None):


    def ffilter(dirs, files):
        files = [f for f in files if 'test' not in f]
        return dirs, files

    X = b.read_dataset(file_path, None, ffilter=ffilter, fread_file=read_file, fprocess=None,
                          show_pbar=True, data_file_deep=3)

    X = X[0]
    if task['target'] == 'rul':
        X = X[X.component.isin(X[X.fault == 1].component.unique())]

    return X
