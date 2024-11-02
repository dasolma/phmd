import pandas as pd
import os
import scipy as scipy
from phmd.readers import base as b

HEADERS = ["vibration"]

FAULTS = ["health_state", "inner_race", "outer_race",]

def read_file(f, file_path, bearing):
    struct = scipy.io.loadmat(f)

    exp = int(os.path.split(file_path)[1].replace('.mat', '').split("_")[-1])
    fault = FAULTS.index(os.path.split(os.path.split(file_path)[0])[-1])

    cols = [t[0] for t in eval(str(struct['bearing'][0].dtype))]

    data = struct['bearing'][0][0]
    X = pd.DataFrame()
    X['vibration'] = data[cols.index('gs')][:, 0]
    X['sr'] = int(data[cols.index('sr')][0, 0])
    X['load'] = int(data[cols.index('load')][0, 0] if len(data[cols.index('load')].shape) == 2 else data[cols.index('load')][0])
    X['rate'] = int(data[cols.index('rate')][0, 0])
    X['fault'] = fault
    X['run'] = int(os.path.split(file_path)[1].replace(".mat", "").split("_")[-1])
    X['unit'] = f"{fault}_{exp}"
    return X



def read_data(file_path, task: dict = None, filters: dict = None):

    X = b.read_dataset(file_path, None, ffilter=None, fread_file=read_file, fprocess=None,
                          show_pbar=True, data_file_deep=3)

    # compute RUL


    return X
