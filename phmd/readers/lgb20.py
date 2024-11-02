import os
import scipy as scipy
from phmd.readers import base as b
import pandas as pd


def read_file(f, file_path, _):
    cycle = pd.read_csv(f, skiprows=30)

    cycle.columns = ['Time Stamp', 'Step', 'Status', 'Prog Time', 'Step Time', 'Cycle',
                     'Cycle Level', 'Procedure', 'Voltage', 'Current', 'Temperature', 'Capacity', 'WhAccu', 'Cnt',
                     'Empty']
    cycle = cycle[(cycle["Status"] == "TABLE") | (cycle["Status"] == "DCH")]

    nominal_capacity = 3
    max_discharge = abs(min(cycle["Capacity"]))
    cycle["SoC Capacity"] = max_discharge + cycle["Capacity"]
    cycle["soc"] = cycle["SoC Capacity"] / nominal_capacity
    X = cycle[["Voltage", "Current", "Temperature", "soc"]].copy()
    T = int(os.path.split(os.path.split(file_path)[0])[1].replace('degC', '').replace('n', ''))
    X['unit'] = (T + float(os.path.split(file_path)[1].replace('.csv', '')[-1]) / 10)
    X['T'] = T

    return X


def read_data(file_path, task: dict = None, filters: dict = None):

    def ffilter(dirs, files):
        files = [f for f in files if 'Mixed' in f]
        return dirs, files

    X = b.read_dataset(file_path, None, ffilter=ffilter, fread_file=read_file, fprocess=None,
                          show_pbar=True, data_file_deep=3)

    X = X[0]
    return X
