import pandas as pd
import scipy as scipy
from phmd.readers import base as b
from collections import defaultdict


STATES = ['C1ch', 'C1dc', 'OCVch', 'OCVdc']

def read_file(f, file_path, _):
    #unit = os.path.split(file_path)[1].replace('.mat', '')

    data = scipy.io.loadmat(f)

    data_dict = defaultdict(lambda: [])
    for cell in list(data.keys())[3:]:
        cell_data = data[cell]

        for i, (cycle, _) in enumerate(eval(str(cell_data.dtype))):
            cycle_data = cell_data[0][0][i]

            for j, (state, _) in enumerate(eval(str(cycle_data.dtype))):
                state_data = cycle_data[0][0][j]

                for k, (feature, _) in enumerate(eval(str(state_data.dtype))):
                    feature_data = list(state_data[0][0][k][:, 0])

                    data_dict[feature] += feature_data

                N = len(feature_data)
                data_dict['state'] += [STATES.index(state)] * N
                #data_dict['cycle'] += [int(cycle[3:])] * N
                data_dict['cell'] += [int(cell[4:])] * N
                data_dict['cycle'] += [int(i)] * N


    X = pd.DataFrame(data_dict)

    r = X.groupby('cell').cycle.max().reset_index()
    r.rename({'cycle': 'max_cycle'}, axis='columns', inplace=True)
    
    X = pd.merge(X, r)

    X['rul'] = X.max_cycle - X.cycle
    del X['max_cycle']
    del X['cycle']
    del X['t']

    return X


def read_data(file_path, task: dict = None, filters: dict = None):

    def ffilter(dirs, files):
        return dirs, files

    X = b.read_dataset(file_path, None, ffilter=ffilter, fread_file=read_file, fprocess=None,
                          show_pbar=True, data_file_deep=3)


    return X
