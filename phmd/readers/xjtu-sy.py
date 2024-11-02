from warnings import filters
import zipfile
import pandas as pd
import os
from typing import List, Dict
import tqdm

HEADERS = ['H_acc', 'V_acc']


def encode_fault(types: list):
    value = [1 if t in types else 0 for t in FAULT_TYPES]

    return value


FAULT_TYPES = ['Outer race', 'Cage', 'Inner race']
BEARING_FAULT_TYPE = {
    (1, 1): ['Outer race'],
    (1, 2): ['Outer race'],
    (1, 3): ['Outer race'],
    (1, 4): ['Cage'],
    (1, 5): ['Inner race', 'Cage'],
    (2, 1): ['Inner race'],
    (2, 2): ['Outer race'],
    (2, 3): ['Cage'],
    (2, 4): ['Outer race'],
    (2, 5): ['Outer race'],
    (3, 1): ['Outer race'],
    (3, 2): ['Outer race', 'Inner race', 'Cage'],
    (3, 3): ['Inner race'],
    (3, 4): ['Inner race'],
    (3, 5): ['Outer race'],
}

BEARING_FAULT_TYPE_ENC = {
    (1, 1): encode_fault(BEARING_FAULT_TYPE[(1, 1)]),
    (1, 2): encode_fault(BEARING_FAULT_TYPE[(1, 2)]),
    (1, 3): encode_fault(BEARING_FAULT_TYPE[(1, 3)]),
    (1, 4): encode_fault(BEARING_FAULT_TYPE[(1, 4)]),
    (1, 5): encode_fault(BEARING_FAULT_TYPE[(1, 5)]),
    (2, 1): encode_fault(BEARING_FAULT_TYPE[(2, 1)]),
    (2, 2): encode_fault(BEARING_FAULT_TYPE[(2, 2)]),
    (2, 3): encode_fault(BEARING_FAULT_TYPE[(2, 3)]),
    (2, 4): encode_fault(BEARING_FAULT_TYPE[(2, 4)]),
    (2, 5): encode_fault(BEARING_FAULT_TYPE[(2, 5)]),
    (3, 1): encode_fault(BEARING_FAULT_TYPE[(3, 1)]),
    (3, 2): encode_fault(BEARING_FAULT_TYPE[(3, 2)]),
    (3, 3): encode_fault(BEARING_FAULT_TYPE[(3, 3)]),
    (3, 4): encode_fault(BEARING_FAULT_TYPE[(3, 4)]),
    (3, 5): encode_fault(BEARING_FAULT_TYPE[(3, 5)]),
}


def read_file(f, file_path, bearing, task):

    X_aux = pd.read_csv(f, delimiter=",")
    X_aux.columns = HEADERS

    bearing = bearing.split('/')[-1]
    condition, number = int(bearing[7]), int(bearing[9])
    run = int(os.path.split(file_path)[1].replace('.csv', ''))

    if task['target'] == 'rul':
        X_aux['unit'] = f"{bearing}_{condition}"
    else:
        X_aux['unit'] = f"{bearing}_{condition}.{run}"
    X_aux['bearing'] = bearing #bearing.split('_')[0]


    target = BEARING_FAULT_TYPE_ENC[(condition, number)]
    for i in range(3):
        X_aux[FAULT_TYPES[i]] = target[i]

    # X_aux['Fault_element'] = BEARING_FAULT_TYPE_ENC[(condition, number)]
    # X_aux['Fault_element'] = X_aux.Fault_element.astype('int8')

    return X_aux


def filter_files(dirs, files, filters: dict = None):
    filtered_files = []
    filtered_dirs = []
    for bearing in dirs:
        bearing_name = bearing.split('/')[-1]
        condition, number = bearing_name[7], bearing_name[9]

        if filters is not None:
            if 'Bearing' in filters and number not in filters['Bearing']:
                continue

            if 'Condition' in filters and condition not in filters['Condition']:
                continue

            if 'Fault_element' in filters and not any(fe in filters['Fault_element'] for fe in fault_element):
                continue

        filtered_dirs.append(bearing)
        bearing_files = sorted([f for f in files if bearing in f])
        filtered_files += bearing_files

        # debug
        #filtered_files = [f for f in filtered_files if '/1.csv' in f]

    return filtered_dirs, filtered_files


def read_files(file_path: str, dirs: List[str], files: List[str], z=None, filters: Dict[str, list] = filters, task=None):
    dirs, files = filter_files(dirs, files, filters)
    datasets = []

    with tqdm.tqdm(total=len(files), desc="Reading dataset") as pbar:

        for bearing in dirs:
            bearing_name = bearing.split('/')[-1]
            pbar.set_description("Reading %s" % bearing_name)

            bearing_files = sorted([f for f in files if bearing in f],
                                   key=lambda x: int(x.split('/')[-1].split('.')[0]))

            ds = []
            for i, bearing_file in enumerate(bearing_files):
                if z is None:
                    with open(os.path.join(file_path, bearing_file), 'r') as f:
                        ds.append(read_file(f, bearing_file, bearing, task))

                else:
                    with z.open(bearing_file) as f:
                        ds.append(read_file(f, bearing_file, bearing, task))

                pbar.update()

            if len(ds) > 0:
                X = pd.concat(ds, axis=0)
                X = X.reset_index(drop=True)

                # compute RUL
                X['RUL'] = (X.index / 32768).astype('int')[::-1].values
                X.RUL = X.RUL.astype('int32')

                datasets.append(X)

    return datasets


def read_dataset(file_path: str, task: dict = None, filters: dict = None):


    if os.path.isdir(file_path):
        dirs = sorted(os.listdir(file_path))
        files = [os.path.join(_dir, file) for _dir in dirs for file in os.listdir(os.path.join(file_path, _dir))]
        datasets = read_files(file_path, dirs, files, filters=filters, task=task)

    else:
        with zipfile.ZipFile(file_path) as z:
            ilen = lambda x: len(list(filter(None, x.split('/'))))
            files = [f for f in z.namelist() if ilen(f) == 4]
            dirs = sorted(set(['/'.join(f.split('/')[:-1]) for f in files]))

            datasets = read_files(file_path, dirs, files, z=z, filters=filters,  task=task)

    X = pd.concat(datasets, axis=0)
    X['H_acc'] = X['H_acc'].astype('float32')
    X['V_acc'] = X['H_acc'].astype('float32')
    X['rul'] = X['RUL'].astype('int32')
    del X['RUL']

    return X
