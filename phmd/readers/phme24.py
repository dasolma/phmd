import os
from phmd.readers import base as b
import pandas as pd
import numpy as np
COLS = ["time", "pos_ref", "pos_fbk", "vel_ref", "vel_fbk", "fbk_dighall", "fbk_digenc1", "drv_prot_vubs",
        "mot_prot_temp", "fbk_cur_a", "fbk_cur_b", "fbk_cur_c", "drv_prot_temp", "fbk_vol_a", "fbk_vol_b", "fbk_vol_c"]

def read_file(f, file_path, _, Xp=None):
    X = pd.read_csv(f, sep=";", names=COLS)

    X["unit"] = int(os.path.split(os.path.split(file_path)[0])[1].replace("Train_", ""))
    file_path = file_path.replace('.csv', '')
    file_path = os.path.split(file_path)[1].split("_")
    X['closing'] = 0
    X['opening'] = 0
    if file_path[-1] == 'Closing':
        X['closing'] = 1
    else:
        X['opening'] = 1

    X['cycle'] = int(file_path[2])
    X = X.iloc[np.arange(0, X.shape[0], 4)]


    return X


def read_data(file_path, task: dict = None, filters: dict = None):


    def ffilter(dirs, files):
        files = [f for f in files if 'RUL' not in f and 'DS_Store' not in f]

        #files = [f for f in files if 'Train_1/' in f or 'Train_2/' in f or 'Train_3/' in f or 'Train_4/' in f or 'Train_5/' in f or 'Train_6/' in f]
        return dirs, files

    if os.path.exists(os.path.join(file_path, 'processed.csv')):
        X = pd.read_csv(os.path.join(file_path, 'processed.csv'))

    else:

        X = b.read_dataset(file_path, None, ffilter=ffilter, fread_file=read_file, fprocess=None,
                              show_pbar=True, data_file_deep=3, concat_in_file=True)

        X = X[0]
        X = X.sort_values(['unit', 'closing', 'cycle'])
        X['rul'] = X.groupby('unit').cycle.transform('max') - X.cycle

        X.to_csv(os.path.join(file_path, 'processed.csv'), index=False)

    return X
