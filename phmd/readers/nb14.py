import pandas as pd
import os
import numpy as np
from phmd.readers import base as b
import scipy.io

from phmd.readers.base import ensure_iterable, in_filename_filter

FEATURES = ["acc1_1", "acc1_2", "acc2_1", "acc2_2", "acc3_1", "acc3_2", "acc4_1", "acc4_2"]

NOMINAL_CAPACITY = 2.2

def to_padded_numpy(l, shape):
    padded_array = np.zeros(shape)
    for i,j in enumerate(l):
        padded_array[i][0:len(j)] = j
    return padded_array

def read_file(f, file_path, filters):
    """
    Base on https://github.com/MichaelBosello/battery-rul-estimation
    """
    struct = scipy.io.loadmat(f)

    raw_data = struct['data'][0][0][0][0]
    cycle = pd.DataFrame(raw_data)

    cycle_x = []
    cycle_y = []
    first_y = True
    y_between_count = 0
    time = []
    current = []
    n_cycles = 0
    max_step = 0

    cycle_num = 0
    cycle['cycle'] = cycle_num
    current_type = cycle.loc[0, 'type']
    for index in range(1, len(cycle.index)):
        if ((current_type == "C" and cycle.loc[index, 'type'] == "D") or
                (current_type == "D" and cycle.loc[index, 'type'] == "C") or
                (current_type == "R" and cycle.loc[index, 'type'] != "R")):
            current_type = cycle.loc[index, 'type']
            cycle_num += 1
        cycle.loc[index, 'cycle'] = cycle_num

    for x in set(cycle["cycle"]):
        if cycle.loc[cycle["cycle"] == x, "type"].iloc[0] != "D":
            continue

        cycle_x.append(np.column_stack([
            np.hstack(cycle.loc[cycle["cycle"] == x, "voltage"].to_numpy().flatten()).flatten(),
            np.hstack(cycle.loc[cycle["cycle"] == x, "current"].to_numpy().flatten()).flatten(),
            np.hstack(cycle.loc[cycle["cycle"] == x, "temperature"].to_numpy().flatten()).flatten()]))

        n_cycles += 1
        step_time = np.hstack(cycle.loc[cycle["cycle"] == x, "time"].to_numpy().flatten()).flatten()
        time.append(step_time / 3600)
        current.append(np.hstack(cycle.loc[cycle["cycle"] == x, "current"].to_numpy().flatten()).flatten())
        max_step = max([max_step, cycle_x[-1].shape[0]])

        if (cycle.loc[cycle["cycle"] == x, "comment"].iloc[0] == "reference discharge" and
                (x < 2 or cycle.loc[cycle["cycle"] == x - 2, "comment"].iloc[0] != "reference discharge")):
            current_y = np.trapz(current[-1], np.hstack(
                cycle.loc[cycle["cycle"] == x, "time"].to_numpy().flatten()).flatten()) / 3600
            if y_between_count > 0:
                step_y = (cycle_y[-1] - current_y) / y_between_count
                while y_between_count > 0:
                    cycle_y.append(cycle_y[-1] - step_y)
                    y_between_count -= 1
            cycle_y.append(current_y)
        elif first_y is True:
            cycle_y.append(NOMINAL_CAPACITY)
        else:
            y_between_count += 1
        first_y = False

    while y_between_count > 0:
        cycle_y.append(cycle_y[-1])
        y_between_count -= 1


    X = pd.DataFrame(np.vstack(cycle_x))
    X.columns = ['voltage', 'current', 'temperature']
    X['rul'] = np.hstack([ [n_cycles - i -1]*cycle_x[i].shape[0] for i in range(n_cycles)])
    X['ahRul']  = np.hstack([[cycle_y[i]]*cycle_x[i].shape[0] for i in range(n_cycles)])
    X['unit'] = os.path.split(file_path)[1].split('.')[0]

    return X


def read_data(file_path, task: dict = None, filters: dict = None):

    def filter_files(dirs, files):
        files = list(filter(lambda x: os.path.split(x)[1].split('.')[0][2:] not in ['9', '10', '11', '12', '20', '3'], files))
        if filters is not None and 'unit' in filters:
            units = ensure_iterable(filters['unit'])
            files = in_filename_filter(units, files)

        return dirs, files

    def __read_file(f, file_path, _):
        return read_file(f, file_path, filters)

    Xs = b.read_dataset(file_path, None, ffilter=filter_files, fread_file=__read_file, fprocess=None,
                       show_pbar=True)

    return Xs
