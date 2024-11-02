import pandas as pd
import os
from phmd.readers import base as b

FEATURES = ["FX", "FY", "FZ", "VX", "VY", "VZ", "AERMS"]

def read_wear(f, file_path, filters = {}, Xp = None):

    X = pd.read_excel(f, skiprows=5)

    X['unit'] = int(os.path.split(os.path.split(file_path)[0])[1][-1])
    X = X.iloc[:, :2]
    X.columns = ['cycle', 'crack']
    X['exp'] = os.path.split(file_path)[1].replace('.xlsx', '').split("_")[1]
    
    return X

def read_features(f, file_path, cracks, filters = {}, Xp = None):

    X = pd.read_csv(f)
    time = float(os.path.split(os.path.split(file_path)[0])[1])
    exp = os.path.split(os.path.split(os.path.split(file_path)[0])[0])[1]
    signal = int(os.path.split(os.path.split(file_path)[1])[1].replace('.csv', '').split('_')[1])

    c = cracks[(cracks.exp == exp)].reset_index(drop=True)
    X['crack'] = c[c.cycle == time].crack.iloc[0]
    X['unit'] = int(exp[1]) * 10 + signal  + (c[c.cycle == time].index[0] / 100)
    X = X[['unit', 'ch1', 'ch2', 'crack']]



    return X


def read_data(file_path, task: dict = None, filters: dict = None):


    def filter_sensor_data(dirs, files):
        files = [f for f in files if '.csv' in f]
        files = [f for f in files if ('T3' not in f) and ('55391' not in f)]
        files = [f for f in files if ('T4' not in f) and ('67054' not in f)]
        files = [f for f in files if ('T4' not in f) and ('67054' not in f)]
        files = [f for f in files if ('Constant' not in f)]
        files = [f for f in files if ('Variable' not in f)]

        return dirs, files

    def filter_crack_data(dirs, files):

        files = [f for f in files if ('Description' in f)]


        return dirs, files

    Ws = b.read_dataset(file_path, None, ffilter=filter_crack_data, fread_file=read_wear, fprocess=None,
                       show_pbar=False, data_file_deep=4, concat_in_file=False)

    cracks = Ws[0]

    def read_features_proxy(f, file_path, filters={}, Xp=None):

        return read_features(f, file_path, cracks)

    Xs = b.read_dataset(file_path, None, ffilter=filter_sensor_data, fread_file=read_features_proxy, fprocess=None,
                       show_pbar=True, data_file_deep=5, concat_in_file=False)

    X = Xs[0]

    return X


def read_train(*args, **kwargs):
    return read_data(*args, **kwargs)


def read_test(*args, **kwargs):
    return read_data(read_wears=False, *args, **kwargs)