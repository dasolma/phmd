import pandas as pd
import os
import numpy as np
from phmd.readers import base as b

COLUMNS = ["time", "u1", "u2", "u3", "u4", "u5", "utot", "j", "i", "tinh2", "touth2", "tinair",
                "toutair", "tinwat", "toutwat", "pinair", "poutair", "pouth2", "pinh2", "dinh2", "douth2", "dinair",
                "doutair", "dwat", "hrairfc"]

def read_file(f, file_path, filters={}):
    unit = os.path.split(file_path)[1].split('_')[0]
    X = pd.read_csv(f, header=None, skiprows=1)

    if X.shape[1] > len(COLUMNS):
        X = X.iloc[:, :len(COLUMNS)]
    X.columns = COLUMNS
    X['unit'] = unit



    return X


def read_data(file_path, task: dict = None, filters: dict = None, params={}):

    Xs = b.read_dataset(file_path, None, ffilter=None, fread_file=read_file, fprocess=None,
                          show_pbar=True, data_file_deep=3)

    def compute_target(X, thresold, name):
        min_u = X.utot[0] - X.utot[0] * thresold
        min_pos = np.where(X.utot <= min_u)[0][0]

        fail_time = X.iloc[min_pos].time
        X[name] = fail_time - X.time
        X.loc[X[name] < 0, name] = 0

        return X


    Xs = [compute_target(X, 0.055, 'rul55') for X in Xs]
    Xs = [compute_target(X, 0.05, 'rul50') for X in Xs]
    Xs = [compute_target(X, 0.045, 'rul45') for X in Xs]
    Xs = [compute_target(X, 0.04, 'rul40') for X in Xs]
    Xs = [compute_target(X, 0.035, 'rul35') for X in Xs]

    return Xs
