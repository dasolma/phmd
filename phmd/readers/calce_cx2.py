from collections import defaultdict

import pandas as pd
import os
import tqdm
import numpy as np
from phmd.readers import base as b
from datetime import datetime as dt
import scipy.io

from phmd.readers.base import ensure_iterable, in_filename_filter

MATERIALS = ["cast iron", "steel"]


def ffill(arr):
    mask = np.isnan(arr)
    idx = np.where(~mask, np.arange(mask.shape[0]), 0)
    np.maximum.accumulate(idx, axis=0, out=idx)
    out = arr[idx]
    return out

def drop_outlier(array, bins):
    index = []
    range_ = np.arange(1, array.shape[0], int(bins/2))
    for i in range_[:-1]:
        array_lim = array[i:i + bins]
        sigma = np.std(array_lim)
        mean = np.mean(array_lim)
        th_max, th_min = mean + sigma * 2, mean - sigma * 2
        idx = np.where((array_lim > th_max) | (array_lim < th_min))[0]
        if idx.shape[0] > 0:
            idx = [iidx + i for iidx in idx]
            array[idx] = np.nan

    return ffill(array)


def read_file(f, file_path, filters={}):
    """
    Based on code: https://github.com/XiuzeZhou/CALCE/blob/main/CALCE.ipynb
    Args:
        f:
        file_path:
        filters:

    Returns:

    """

    ext = os.path.splitext(file_path)[1]
    if ext == '.txt':
        X = pd.read_csv(f, sep='\t')
    elif ext == '.xlsx':
        xl = pd.ExcelFile(file_path)
        sheets = []
        for nsheet in xl.sheet_names:
            try:
                X = pd.read_excel(f, sheet_name=nsheet)

                if 'Date_Time' in X.columns:
                    sheets.append(X)
            except:
                print("Error in file %s" % file_path)

        X = pd.concat(sheets, axis=0)
        print(X.shape)
        X.to_csv(file_path.replace('.xlsx', '.csv'), index=False)
    elif ext == '.csv':
        X = pd.read_csv(f)

    X['Date_Time'] = pd.to_datetime(X.Date_Time)
    X['timestamp'] = X.Date_Time.astype(int)

    X = X.sort_values('timestamp')
    X = X.reset_index(drop=True)

    # constant current = 2 (charging), constant voltage = 4 (charging), discharging = 7
    X = X[X.Step_Index.isin([2, 4, 7])]

    file_name_parts = os.path.split(os.path.splitext(file_path)[0])[1].split('_')
    unit = int(os.path.split(os.path.split(file_path)[0])[1].split('_')[1])
    X['unit'] = unit

    #

    Xd = X[X.Step_Index == 7]
    capacities = defaultdict(lambda: [])
    for cycle in Xd.Cycle_Index.unique():
        time_diff = np.diff(Xd[Xd.Cycle_Index == cycle]['Test_Time(s)'].values)
        current = Xd[Xd.Cycle_Index == cycle]['Current(A)'].values[1:]

        # Q = A * h
        capacities['capacity'].append(-1 * np.sum(time_diff * current / 3600))
        capacities['unit'] = unit
        capacities['timestamp'] = Xd[Xd.Cycle_Index == cycle]['timestamp'].values[-1]

    if filters is not None and 'mode' in filters:
        modes = ensure_iterable(filters['mode'])
        _modes = []
        if 'charging' in modes:
            modes = [2, 4]

        if 'discharging' in modes:
            modes.append(7)

        X = X[X.Step_Index.isin(_modes)]

    return X, pd.DataFrame(capacities)


def read_data(file_path, task: dict = None, filters: dict = None, params={}):
    def __read_file(f, file_path, _):
        return read_file(f, file_path, filters)

    def ffilter(dirs, files):
        files = [f for f in files if os.path.splitext(f)[1] == '.csv']
        #files = [f for f in files if os.path.splitext(f)[1] == '.xlsx' and f.replace('xlsx', 'csv') not in files]


        if filters is not None and 'unit' in filters:
            units = ensure_iterable(filters['unit'])
            files = in_filename_filter(units, files)
        else:
            units = ["CX2_33", "CX2_34", "CX2_35", "CX2_36", "CX2_37", "CX2_38"]
            files = in_filename_filter(units, files)


        return dirs, files


    X, C = b.read_dataset(file_path, None, ffilter=ffilter, fread_file=__read_file, fprocess=None,
                       show_pbar=True, data_file_deep=5)[0]


    C = C.sort_values(['unit', 'timestamp'])
    C = C.reset_index(drop=True)
    C = C[C.capacity > 0]
    for unit in C.unit.unique():
        C.loc[C.unit == unit, 'capacity'] = drop_outlier(C.loc[C.unit == unit, 'capacity'].values, 40)
        C.loc[C.unit == unit, 'cycle'] = np.arange(0, C.loc[C.unit == unit].shape[0])
        C.loc[C.unit == unit, 'soc'] = C.loc[C.unit == unit, 'capacity'] / 1.35

        if task is not None:
            if task.get('target', None) == 'soc_ahead':
                i = params['soc_estimation_ahead_cycles']
                C.loc[C.unit == unit, 'soc_ahead'] = np.concatenate((C.loc[C.unit == unit].soc.values[:-i], [None]*i))

            elif task.get('target', None) == 'rul':
                rul = np.arange(C.loc[C.unit == unit].shape[0]-1, -1, -1)
                thr = params['rul_soc_threshold']
                mask = np.where(C.loc[(C.unit == unit), 'soc'] < thr)[0]
                if mask.shape[0] > 0:
                    ifail = np.where(C.loc[(C.unit == unit), 'soc'] < thr)[0][0]
                    rul[ifail:] = 0
                    rul[:ifail] -= rul[:ifail].min() - 1
                else:
                    rul -= rul.min()
                C.loc[C.unit == unit, 'rul'] = rul

    X = pd.merge(X, C, on=['unit', 'timestamp'], how='left')
    X = X.sort_values(['unit', 'timestamp'])
    X = X.reset_index(drop=True)

    if task is not None:
        task_name = task.get('target', None)
        if task_name == 'soc_ahead':
            C = C[~C.soc_ahead.isnull()]

        # SOC filling
        for unit in C.unit.unique():
            X.loc[X.unit == unit, 'soc'] = X.loc[X.unit == unit].soc.bfill()

            if task_name == 'soc_ahead':
                X.loc[X.unit == unit, 'soc_ahead'] = X.loc[X.unit == unit].soc_ahead.bfill()

        if task_name == 'soc_ahead':
            X = X[~X.soc_ahead.isnull()]

        elif task_name == 'rul':
            for unit in C.unit.unique():
                X.loc[X.unit == unit] = X.loc[X.unit == unit].ffill()

            X = X[~X.rul.isnull()]

    return X
