from collections import defaultdict
import pandas as pd
import os
import numpy as np
from phmd.readers import base as b
from datetime import datetime as dt
import scipy.io

#todo: review warnings

from phmd.readers.base import ensure_iterable, in_filename_filter

FEATURES = ["acc1_1", "acc1_2", "acc2_1", "acc2_2", "acc3_1", "acc3_2", "acc4_1", "acc4_2"]


def ensure_length(data):
    max_len = max([len(data[k]) for k in data.keys()])
    if not all([len(data[k]) == max_len for k in data.keys()]):
        __ensure_length = lambda data_list, n: data_list + [None] * (n - len(data_list))
        data = {k: __ensure_length(data[k], max_len) for k in data.keys()}
    return data



def build_dictionaries(mess, filters):
    """
    Based on:  https://notebook.community/adrienBizeray/battery_skunkworks/nasa-battery-data-analysis
    Args:
        mess:

    Returns:

    """
    #discharge, charge, impedance = defaultdict(lambda: []), defaultdict(lambda: []), defaultdict(lambda: [])
    alldata = defaultdict(lambda: [])
    cols = ['mode', 'cycle', 'amb_temp', 'voltage_battery', 'current_battery',
       'temp_battery', 'current_charge', 'voltage_charge',
       'sense_current_real', 'sense_current_imag', 'battery_current_real',
       'battery_current_imag', 'current_ratio_real', 'current_ratio_imag',
       'battery_impedance_real', 'battery_impedance_imag',
       'rectified_impedance_real', 'rectified_impedance_imag', 're', 'rct']
    modes = ["discharge", "charge", "impedance"]
    if filters is not None:
        modes = ensure_iterable(["discharge", "charge", "impedance"] if 'mode' not in filters else filters['mode'])

    def add_zeros():
        L = max([len(alldata[c]) for c in alldata.keys()])

        for c in alldata.keys():
            if len(alldata[c]) < L:
                alldata[c] = alldata[c] + ([0] * (L - len(alldata[c])))
        
    sets = []
    for i, element in enumerate(mess):

        step = element[0][0]

        year = int(element[2][0][0])
        month = int(element[2][0][1])
        day = int(element[2][0][2])
        hour = int(element[2][0][3])
        minute = int(element[2][0][4])
        second = int(element[2][0][5])
        millisecond = int((second % 1) * 1000)
        date_time = dt(year, month, day, hour, minute, second, millisecond)
        date_time = date_time.strftime("%d %b %Y, %H:%M:%S")


        if step == 'discharge' and 'discharge' in modes:
            data = element[3]
            ndata = len(data[0][0][1][0].tolist())

            discharge = defaultdict(lambda: [])
            discharge['mode'] += [0] * ndata
            discharge['cycle'] += [i] * ndata
            discharge["amb_temp"] += [str(element[1][0][0])] * ndata
            discharge['step'] = [1] * ndata
            #discharge["date_time"] += [date_time] * ndata

            discharge["voltage_battery"] += data[0][0][0][0].tolist()
            discharge["current_battery"] += data[0][0][1][0].tolist()
            discharge["temp_battery"] += data[0][0][2][0].tolist()
            discharge["current_charge"] += data[0][0][3][0].tolist()
            discharge["voltage_charge"] += data[0][0][4][0].tolist()
            #discharge["time"] += data[0][0][5][0].tolist()

            discharge = ensure_length(discharge)

            ndata = len(discharge['mode'])

            #add_zeros()
            #assert all([len(discharge[k]) == ndata for k in discharge.keys()])

            sets.append(pd.DataFrame(discharge))

        if step == 'charge' and 'charge' in modes:
            data = element[3]
            ndata = len(data[0][0][1][0].tolist())

            charge = defaultdict(lambda: [])
            charge['mode'] += [1] * ndata
            charge["amb_temp"] += [str(element[1][0][0])] * ndata
            charge['cycle'] += [i] * ndata
            charge['step'] = [0] * ndata

            charge["voltage_battery"] += data[0][0][0][0].tolist()
            charge["current_battery"] += data[0][0][1][0].tolist()
            charge["temp_battery"] += data[0][0][2][0].tolist()
            charge["current_charge"] += data[0][0][3][0].tolist()
            charge["voltage_charge"] += data[0][0][4][0].tolist()

            charge = ensure_length(charge)

            ndata = len(charge['mode'])

            #add_zeros()
            #assert all([len(charge[k]) == ndata for k in charge.keys()])

            sets.append(pd.DataFrame(charge))

        if step == 'impedance' and 'impedance' in modes:
            data = element[3]
            ndata = len(data[0][0][0][0].tolist())

            impedance = defaultdict(lambda: [])
            impedance['mode'] += [2] * ndata
            impedance['cycle'] += [i] * ndata
            impedance['step'] = [2] * ndata

            impedance["amb_temp"] += [str(element[1][0][0])] * ndata
            #impedance["date_time"] += [date_time] * ndata

            impedance["sense_current_real"] += np.real(data[0][0][0][0]).tolist()
            impedance["sense_current_imag"] += np.imag(data[0][0][0][0]).tolist()

            impedance["battery_current_real"] += np.real(data[0][0][1][0]).tolist()
            impedance["battery_current_imag"] += np.imag(data[0][0][1][0]).tolist()

            impedance["current_ratio_real"] += np.real(data[0][0][2][0]).tolist()
            impedance["current_ratio_imag"] += np.imag(data[0][0][2][0]).tolist()

            impedance["battery_impedance_real"] += np.real(data[0][0][3][0]).tolist()
            impedance["battery_impedance_imag"] += np.imag(data[0][0][3][0]).tolist()

            impedance["rectified_impedance_real"] += np.real(data[0][0][4][0]).tolist()
            impedance["rectified_impedance_imag"] += np.imag(data[0][0][4][0]).tolist()

            impedance["re"] += [float(data[0][0][5][0][0])] * ndata
            impedance["rct"] += [float(data[0][0][6][0][0])] * ndata

            impedance = ensure_length(impedance)

            ndata = len(impedance['mode'])

            #add_zeros()
            #assert all([len(impedance[k]) == ndata for k in impedance.keys()])

            sets.append(pd.DataFrame(impedance))

    return pd.concat(sets, axis=0)


def read_file(f, file_path, filters):
    unit = os.path.splitext(os.path.split(file_path)[1])[0]
    struct = scipy.io.loadmat(f)
    mess = struct[unit][0][0][0][0]
    data = build_dictionaries(mess, filters)

    #data = pd.concat(data, axis=0)
    data['unit'] = unit
    data = data.ffill()
    data = data.fillna(0)

    return data


def read_data(file_path, task: dict = None, filters: dict = None):

    def filter_files(dirs, files):
        if filters is not None and 'unit' in filters:
            units = ensure_iterable(filters['unit'])
            files = in_filename_filter(units, files)

        return dirs, files

    def __read_file(f, file_path, _):
        return read_file(f, file_path, filters)

    Xs = b.read_dataset(file_path, None, ffilter=filter_files, fread_file=__read_file, fprocess=None,
                       show_pbar=True)

    X = Xs[0]
    max_seconds_per_unit = X.groupby('unit')['cycle'].max().reset_index()
    X = X.merge(max_seconds_per_unit, how='left', on='unit')
    X['rul'] = (X.cycle_y - X.cycle_x)

    del X['cycle_x']
    del X['cycle_y']

    return X

