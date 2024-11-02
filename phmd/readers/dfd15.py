import pandas as pd
import os
import numpy as np
from phmd.readers import base as b
import re

FAULTS = ["flank wear", "chisel wear", "outer corner wear", "perfect"]

def read_file(f, file_path, filters):

    fault = int(os.path.split(os.path.split(file_path)[0])[1].split('_')[1]) -1
    stage = len(os.path.split(file_path)[1].split('-')) - 1
    data = np.array([float(d) for d in f.readlines()])
    try:
        unit = re.search(r"\d+(?:[\.\-]\d+)?", os.path.split(file_path)[1]).group()
        unit = str(fault) + "_" + unit
    except:
        print("ok")


    X = pd.DataFrame({'vibration': data})
    X['fault'] = fault
    X['stage'] = stage
    X['unit'] = unit

    return X


def read_data(file_path, task: dict = None, filters: dict = None):
    def filter_files(dirs, files):
        return dirs, files


    def __read_file(f, file_path, _):
        return read_file(f, file_path, filters)


    Xs = b.read_dataset(file_path, None, ffilter=filter_files, fread_file=__read_file, fprocess=None, join=None,
                        show_pbar=True, data_file_deep=4)

    X = Xs[0]

    X = X[X.unit != '0_1']


    return X
