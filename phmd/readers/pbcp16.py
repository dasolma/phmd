import pandas as pd
import os
from phmd.readers import base as b

FAULTS = ["normal", "roller wearing", "inner race wearing", "outer race wearing", "impeller wearing"]

def read_file(f, file_path, bearing):
    experiment = int(os.path.split(file_path)[1].replace("ABC", "").split('#')[0])
    fault = os.path.split(os.path.split(file_path)[0])[1]
    fault = [i for i, f in enumerate(FAULTS) if f in fault][0]

    data = f.read()
    if isinstance(data, bytes):
        data = data.decode()
    data = data.strip().split("\n")
    data = [float(v) for v in data[3:]]

    X = pd.DataFrame({'vibration': data})
    X['experiment'] = f"{experiment}_{fault}"
    X['fault'] = fault

    return X



def read_data(file_path, task: dict = None, filters: dict = None):


    def ffilter(dirs, files):
        return dirs, files

    X = b.read_dataset(file_path, None, ffilter=ffilter, fread_file=read_file, fprocess=None,
                          show_pbar=True, data_file_deep=5)

    X = X[0]
    return X
