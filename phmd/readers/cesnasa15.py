import pandas as pd
from phmd.readers import base as b
import numpy as np
import h5py
from collections import defaultdict

COLUMNS = ['time', 'supplyVoltage', 'packageTemperature', 'drainSourceVoltage', 'drainCurrent', 'flangeTemperature']


#todo: review warnings
def extract_header_info(data):

    headers = []
    for i in range (data['ES10']['EIS_Data']['ES10C3']['EIS_Measurement']['Header'].shape[0]):
        for j in range(data[data['ES10']['EIS_Data']['ES10C3']['EIS_Measurement']['Header'][i,0]].shape[0]):
            ascii = data[data[data['ES10']['EIS_Data']['ES10C3']['EIS_Measurement']['Header'][0,0]][0,0]][:]
            txt = '\n'.join([''.join([chr(x) for x in ascii[:, i]]) for i in range(ascii.shape[1])])
            headers.append(txt)
    
    assert len(set(headers)) == 1

    return headers[0]

    
def read_file(f, file_path, _):

    data = h5py.File(f, 'r')
    #data = mat73.loadmat(f)

    batt = [k for k in data.keys() if k != '#refs#'][0]

    X = defaultdict(lambda: [])
    units = [u for u in data[batt]['EIS_Data'].keys() if u != 'EIS_Reference_Table']
    for unit in units:
        ref = data[batt]['EIS_Data'][unit]['EIS_Measurement']['ColumNames'][1, 0]
        col_names = [''.join([chr(x) for x in data[ref][:][:, i]]).strip() for i in range(data[ref][:].shape[0])]

        for i in range(2, 68):

            for j, c in enumerate(col_names):
                d = data[data[data[batt]['EIS_Data'][unit]['EIS_Measurement']['Data'][i, 0]][0, 0]][j][8:]
                X[c].append(d)
                l = len(d)
                #plt.scatter(R, I)

            X['measure'].append(np.array([i] * l))
            X['unit'].append(np.array([unit] * l))

    X = pd.DataFrame({k: np.hstack(v) for k, v in X.items()})


    return X

RDSON_THRESOLD = 0.05

def read_data(file_path, task: dict = None, filters: dict = None):

    def ffilter(dirs, files):
        return dirs, files

    X = b.read_dataset(file_path, None, ffilter=ffilter, fread_file=read_file, fprocess=None,
                          show_pbar=True, data_file_deep=3)

    X = X[0]

    C0 = 0.000312
    X['EweLoss'] =  1 - ((C0 - X['<Ewe>/V']) / C0)
    loss = X.groupby(['unit', 'measure'])['EweLoss'].max().to_dict()
    fail = (X.groupby(['unit', 'measure'])['EweLoss'].max() < 0.5).reset_index()
    fail_measures = fail[fail.EweLoss].groupby('unit').measure.min().reset_index()
    fail_measures.columns = ['unit', 'fail_measure']
    X = pd.merge(X, fail_measures, on='unit')
    X['rul1'] = (X.fail_measure - X.measure).clip(0, 100)

    del X['EweLoss']
    del X['fail_measure']


    def slope(x, y):
        X = x - x.mean()
        Y = y - y.mean()

        slope = (X.dot(Y)) / (X.dot(X))

        return slope

    slopes = X.groupby(['unit', 'measure'])[['Re(Z)/Ohm', '-Im(Z)/Ohm']].apply(
        lambda x: slope(x['Re(Z)/Ohm'].values, x['-Im(Z)/Ohm'].values)).reset_index()
    slopes.columns = ['unit', 'measure', 'slope']
    minmax = slopes.groupby('unit').slope.agg(['min', 'max'])
    thresolds = (minmax['min'] +  (minmax['max'] - minmax['min']) * 0.8).reset_index()
    thresolds.columns = ['unit', 'thresold']
    thresolds = pd.merge(slopes, thresolds)
    thresolds['fail'] = (thresolds.slope > thresolds.thresold)
    fail_measures = thresolds[thresolds.fail].groupby('unit').measure.min().reset_index()
    fail_measures.columns = ['unit', 'fail_measure']
    X = pd.merge(X, fail_measures, on='unit')
    X['rul2'] = (X.fail_measure - X.measure).clip(0, 100)

    del X['fail_measure']
    del X['number']
    del X['cycle']
    del X['time/s']
    del X['measure']

    return X
