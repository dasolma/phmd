import pandas as pd
import os
from phmd.readers import base as b

FAULTS = ['bearing', 'flywheel', 'healthy', 'liv', 'lov', 'nrv', 'piston',
       'riderbelt']

def read_file(f, file_path, bearing):

    fault = FAULTS.index(os.path.split(os.path.split(file_path)[0])[1].lower())
    unit = int(os.path.split(file_path)[1].replace('.dat', '').replace('preprocess_Reading', ''))
    data = f.read().strip()
    if isinstance(data, bytes):
        data = data.decode()
    data = [float(v) for v in data.split(",")]

    X = pd.DataFrame({'acoustic': data})
    X['fault'] = fault
    X['unit'] = fault * 1000 + unit
    #X['batch'] = [i for i in range(X.shape[0]//250) for _ in range(250)][:X.shape[0]]

    return X



def read_data(file_path, task: dict = None, filters: dict = None):

    def ffilter(dirs, files):
        return dirs, files



    X = b.read_dataset(file_path, None, ffilter=ffilter, fread_file=read_file, fprocess=None,
                          show_pbar=True, data_file_deep=4)

    X = X[0]

    return X
