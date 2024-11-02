from collections import defaultdict
import scipy
from phmd.readers import base as b
import pandas as pd
import numpy as np
import os

TYPES = {'K001': ('Healthy', 'Normal'), 'K002': ('Healthy', 'Normal'), 'K003': ('Healthy', 'Normal'),
         'K004': ('Healthy', 'Normal'), 'K005': ('Healthy', 'Normal'), 'K006': ('Healthy', 'Normal'),
         'KA01': ('Artificial', 'OR'), 'KA03': ('Artificial', 'OR'), 'KA05': ('Artificial', 'OR'),
         'KA06': ('Artificial', 'OR'), 'KA07': ('Artificial', 'OR'), 'KA09': ('Artificial', 'OR'),
         'KI01': ('Artificial', 'IR'), 'KI03': ('Artificial', 'IR'), 'KI05': ('Artificial', 'IR'),
         'KI07': ('Artificial', 'IR'), 'KI08': ('Artificial', 'IR'), 'KA04': ('Real', 'OR'),
         'KA15': ('Real', 'OR'), 'KA16': ('Real', 'OR'), 'KA22': ('Real', 'OR'), 'KA30': ('Real', 'OR'),
         'KB23': ('Real', 'IR'), 'KB24': ('Real', 'IR'), 'KI04': ('Real', 'IR'), 'KI14': ('Real', 'IR'),
         'KI16': ('Real', 'IR'), 'KI17': ('Real', 'IR'), 'KI18': ('Real', 'IR'), 'KI21': ('Real', 'IR'),
         'KB27': ('Real', 'OR')}

UNIT_LEFT = ["K0", "KA", "KI", "KB"]

COMPONENT = ['Normal', 'IR', 'OR']
DTYPE = ['Healthy', 'Artificial', 'Real']


def repeat_values(data, max_size):
    repeats = int(max_size / len(data))
    if repeats > 1:
        #data = [r for v in data for r in [v] * repeats]
        data = np.repeat(data, repeats)

    if len(data) < max_size:
        data = np.concatenate((data, [data[-1]] * (max_size - len(data))))

    return data


def read_file(f, file_path, bearing, Xp = None):

    X = scipy.io.loadmat(f)

    name = list(X.keys())[-1]

    unit = name.split("_")[-2]
    rotational_speed, load_torque, radial_face = [float(f[1:]) for f in name.split('_')[:3]]
    run = int(name.split("_")[-1])

    X = X[name][0, 0]

    _type, _component = TYPES[unit]

    max_size = max([d[2].shape[1] for d in X[2][0]])


    data = defaultdict(lambda: np.array([]))
    for d in X[2][0]:
        colname = d[0][0]
        values = d[2][0]


        if len(values) < max_size:
            values = repeat_values(values, max_size)

        data[colname] = np.concatenate((data[colname], values))

    X = pd.DataFrame(data)
    X['unit'] = int(UNIT_LEFT.index(unit[:2])) + int(unit[2:]) / 100
    X['run'] = run
    X['run'] = X.run.astype('int8')

    _type, _component = TYPES[unit]
    X['type'] = DTYPE.index(_type)
    X['type'] = X.type.astype('int8')
    X['fault_component'] = COMPONENT.index(_component)
    X['fault_component'] = X.fault_component.astype('int8')
    X['rotational_speed'], X['load_torque'], X['radial_face'] =  rotational_speed, load_torque, radial_face
    X['rotational_speed'] = X.rotational_speed.astype('float16')
    X['load_torque'] = X.load_torque.astype('float16')
    X['radial_face'] = X.radial_face.astype('float16')
    X['force'] = X.force.astype('float16')
    X['phase_current_1'] = X.phase_current_1.astype('float16')
    X['phase_current_2'] = X.phase_current_2.astype('float16')
    X['speed'] = X.speed.astype('float16')
    X['temp_2_bearing_module'] = X.temp_2_bearing_module.astype('float16')
    X['torque'] = X.torque.astype('float16')


    return X


def read_data(file_path: str, task: dict = None, filters: dict = None):

    def ffilter(dirs, files):
        files = list(filter(lambda f: '.pdf' not in f, files))
        files = list(filter(lambda f: 'KA08' not in f, files))  # missing types of this experiment

        files = sorted(files, key=lambda f: (''.join(f.split('_')[:-1]), int(f.split('_')[-1].replace('.mat', ''))) )
        files = list(filter(lambda f: '.pdf' not in f, files))

        # for debug
        def number_file(file):
            return int(os.path.split(file)[1].replace('.mat', '').split('_')[-1])

        files = [f for f in files if number_file(f) % 3 == 0]

        return dirs, files

    X = b.read_dataset(file_path, None, ffilter=ffilter, fread_file=read_file, fprocess=None,
                       show_pbar=True, data_file_deep=4, sort_files=False, concat_delegated=True,
                       concat_in_file=True)

    X = X[0]

    # split healthy units
    real_units = list(X[(X['type'] == DTYPE.index('Real')) | (X['type'] == DTYPE.index('Healthy'))].unit.unique())
    test_units = X[['fault_component', 'unit']].drop_duplicates().groupby('fault_component').apply(lambda x: x.sample(1)).reset_index(drop=True).unit.values
    train_units = set(real_units) - set(test_units)

    X.rename({'fault_component': 'fault'}, axis='columns', inplace=True)

    # create train and test sets
    X_test = X[X['unit'].isin(test_units)]
    X_train = pd.concat((X[X['unit'].isin(train_units)],
                         X[X['type'] == DTYPE.index('Artificial')]))
    return (X_train, X_test)
    
