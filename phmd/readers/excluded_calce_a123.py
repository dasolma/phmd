"""
Excluded... Implementation unfinished
"""
from collections import defaultdict

import pandas as pd
import os
import numpy as np
from phmd.readers import base as b
from phmd.readers.base import ensure_iterable, in_filename_filter


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
    """

    X = pd.read_excel(f, sheet_name=1)
    assert 'Date_Time' in X.columns

    X['Date_Time'] = pd.to_datetime(X.Date_Time)
    X['timestamp'] = X.Date_Time.astype(int)

    X = X.sort_values('timestamp')
    X = X.reset_index(drop=True)

    file_name_parts = os.path.split(os.path.splitext(file_path)[0])[1].split('_')
    unit = int(os.path.split(os.path.split(file_path)[0])[1].split('_')[1])
    X['unit'] = unit

    #

    step_indexes = X.groupby('Step_Index')['Current(A)'].max().reset_index()
    discharge_step_indexes = step_indexes[step_indexes['Current(A)'] < 0]
    discharge_step_indexes = discharge_step_indexes.Step_Index.unique()
    Xd = X[X.Step_Index.isin(discharge_step_indexes)]
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

    X, C = b.read_dataset(file_path, None, ffilter=None, fread_file=__read_file, fprocess=None,
                       show_pbar=True, data_file_deep=2)


    C = C.sort_values(['unit', 'timestamp'])
    C = C.reset_index(drop=True)
    C = C[C.capacity > 0]
    for unit in C.unit.unique():
        C.loc[C.unit == unit, 'capacity'] = drop_outlier(C.loc[C.unit == unit, 'capacity'].values, 40)
        C.loc[C.unit == unit, 'cycle'] = np.arange(0, C.loc[C.unit == unit].shape[0])
        C.loc[C.unit == unit, 'soc'] = C.loc[C.unit == unit, 'capacity'] / 1.35

        if task is not None:
            if task.get('task_name', None) == 'soc_estimation':
                i = params['soc_estimation_ahead_cycles']
                C.loc[C.unit == unit, 'soc_ahead'] = np.concatenate((C.loc[C.unit == unit].soc.values[:-i], [None]*i))

            elif task.get('task_name', None) == 'rul':
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
        task_name = task.get('task_name', None)
        if task_name == 'soc_estimation':
            C = C[~C.soc_ahead.isnull()]

        # SOC filling
        for unit in C.unit.unique():
            X.loc[X.unit == unit, 'soc'] = X.loc[X.unit == unit].soc.bfill()

            if task_name == 'soc_estimation':
                X.loc[X.unit == unit, 'soc_ahead'] = X.loc[X.unit == unit].soc_ahead.bfill()


        if task_name == 'soc_estimation':
            X = X[~X.soc_ahead.isnull()]
        elif task_name == 'rul':
            pass

        # set task columns
        target = ensure_iterable(task['target'])
        _id = ensure_iterable(task['identifier'])
        columns = list(set(_id  + task['features'] + target))

        return X[columns]
    else:
        return X
