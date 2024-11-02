import pandas as pd
import numpy as np
from phmd.readers import base as b


def read_file(f, file_path, filters):
    data = np.array([np.array([float(d) for d in l.split()]) for l in f.readlines()])
    X = pd.DataFrame(data, columns=['lp', 'ss', 'gtst', 'gtror', 'ggror', 'spt', 'ppt', 'htet', 'gciat', 'gcoat', 'htep', 'gciap', 'gcoap', 'gtegp', 'tic', 'ff', 'gcdsc', 'gtdsc'])
    X['unit'] = 1
    return X


def read_data(file_path, task: dict = None, filters: dict = None):

    def filter_files(dirs, files):
        return dirs, files


    def __read_file(f, file_path, _):
        return read_file(f, file_path, filters)

    Xs = b.read_dataset(file_path, None, ffilter=filter_files, fread_file=__read_file, fprocess=None, join=None,
                        show_pbar=True, data_file_deep=3)

    X = Xs[0]
    return X
