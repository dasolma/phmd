import pandas as pd
import os
from phmd.readers import base as b


PROCESSES = ['starting', 'prep', 'layer 1 up', 'layer 1 down', 'repositioning', 'layer 2 up', 'layer 2 down',
             'layer 3 up', 'layer 3 down', 'end']

def read_file(f, file_path, W):

    unit = int(os.path.split(file_path)[1].split(".")[0].split("_")[1])
    X = pd.read_csv(f)

    X['unit'] = unit
    X['wear'] = 0 if W[W.No==unit].tool_condition.values[0] == 'unworn' else 1
    X['Machining_Process'] = X.Machining_Process.map(lambda x: PROCESSES.index(x.lower()))

    return X

def read_wear(f, file_path, _):

    X = pd.read_csv(f)

    return X

def read_data(file_path, task: dict = None, filters: dict = None):

    def filter_wear_data(dirs, files):
        files = [f for f in files if 'train.csv' in f]

        return dirs, files

    W = b.read_dataset(file_path, None, ffilter=filter_wear_data, fread_file=read_wear, fprocess=None,
                       show_pbar=False, data_file_deep=2)[0]

    def filter_experiments(dirs, files):
        files = [f for f in files if 'experiment' in f]

        return dirs, files

    def read_file_proxy(f, file_path, _):
        return read_file(f, file_path, W)

    X = b.read_dataset(file_path, None, ffilter=filter_experiments, fread_file=read_file_proxy, fprocess=None,
                          show_pbar=True, data_file_deep=2)

    # compute RUL


    return X
