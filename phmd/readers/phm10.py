import pandas as pd
import os
from phmd.readers import base as b
from phmd.readers.base import ensure_iterable, in_filename_filter

FEATURES = ["FX", "FY", "FZ", "VX", "VY", "VZ", "AERMS"]

def read_wear(f, file_path, filters = {}, Xp = None):

    X = pd.read_csv(f)
    X['unit'] = int(os.path.split(os.path.split(file_path)[0])[1][-1])

    return X

def read_features(f, file_path, wears, filters = {}, Xp = None):

    X = pd.read_csv(f, names=FEATURES)
    X['cut'] = int(os.path.split(os.path.splitext(file_path)[0])[1].split('_')[-1])
    X['unit'] = int(os.path.split(os.path.split(file_path)[0])[1][-1])

    X = pd.merge(X, wears, on=['unit', 'cut'], how='left')

    return X



def read_data(file_path, task: dict = None, filters: dict = None):


    def ffilter(dirs, files):
        # debug
        #files = [f for f in files if ('wear' in f) or ('_001.csv' in f)]

        if filters is not None and 'unit' in filters:
            units = ensure_iterable(filters['unit'])
            files = in_filename_filter(['/c' + str(f) for f in units], files)
        return dirs, files
    
    def filter_sensor_data(dirs, files):

        dirs, files = ffilter(dirs, files)

        files = [f for f in files if 'wear' not in f]

        return dirs, files

    def filter_wear_data(dirs, files):

        dirs, files = ffilter(dirs, files)

        files = [f for f in files if 'wear' in f]

        return dirs, files

    Ws = b.read_dataset(file_path, None, ffilter=filter_wear_data, fread_file=read_wear, fprocess=None,
                       show_pbar=False, data_file_deep=5, concat_in_file=True)

    wears = Ws[0]
    wears['wear'] = wears[["flute_1", "flute_2", "flute_3"]].mean(axis=1)
    wears = wears[['unit', 'cut', 'wear']]

    def read_features_proxy(f, file_path, filters={}, Xp=None):

        return read_features(f, file_path, wears)

    Xs = b.read_dataset(file_path, None, ffilter=filter_sensor_data, fread_file=read_features_proxy, fprocess=None,
                       show_pbar=True, data_file_deep=6, concat_in_file=True)


    """
    for i in range(len(Xs)):
        X = Xs[i]

        W = Ws[i]
        W['wear'] = W[["flute_1", "flute_2", "flute_3"]].mean()
        W = W[['unit', 'cut', 'wear']]
        X = pd.merge(X, W, on=['unit', 'cut'])

        Xs[i] = X
    """

    X = Xs[0]
    X['VY'] = X.groupby(['cut', 'unit', 'wear']).VY.ffill()
    X['VZ'] = X.groupby(['cut', 'unit', 'wear']).VZ.ffill()
    X['AERMS'] = X.groupby(['cut', 'unit', 'wear']).AERMS.ffill()

    return X


def read_train(*args, **kwargs):
    return read_data(*args, **kwargs)


def read_test(*args, **kwargs):
    return read_data(read_wears=False, *args, **kwargs)