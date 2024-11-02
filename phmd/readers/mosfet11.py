import pandas as pd
import os
import scipy as scipy
from phmd.readers import base as b
import numpy as np
from dateutil import parser

COLUMNS = ['time', 'supplyVoltage', 'packageTemperature', 'drainSourceVoltage', 'drainCurrent', 'flangeTemperature']


def read_file(f, file_path, _):

    struct = scipy.io.loadmat(f)

    file_name = os.path.split(file_path)[1]
    parts = file_name.replace('.mat', '').split('_')

    device = int(parts[1])
    run = int(parts[3])


    X = struct
    data = np.array([ [r[0][0]] + [d[0] if isinstance(d[0], str) else float(d[0][0]) for d in r[3][0, 0]] for r in X['measurement'][0, 0][2][0] ])
    data = pd.DataFrame(data=data)
    data.columns = COLUMNS
    data['time'] = data.time.apply(lambda x: int(parser.parse(x).timestamp()))
    data['supplyVoltage'] = data.supplyVoltage.astype('float')
    data['packageTemperature'] = data.packageTemperature.astype('float')
    data['drainSourceVoltage'] = data.drainSourceVoltage.astype('float')
    data['drainCurrent'] = data.drainCurrent.astype('float')
    data['flangeTemperature'] = data.flangeTemperature.astype('float')
    #data.set_index('time', inplace=True)


    rds = []
    time = []
    raw_time = []
    idl = []
    vdsl = []
    vsignall = []
    vsourcel = []
    for i in range(len(X['measurement'][0, 0][1][0])):
        gated = X['measurement'][0, 0][1][0, i][3][0][0][1][0, :]
        id = X['measurement'][0, 0][1][0, i][3][0][0][4][0, :]
        vds = X['measurement'][0, 0][1][0, i][3][0][0][3][0, :]
        vsignal = X['measurement'][0, 0][1][0, i][3][0][0][1][0, :]
        vsource = X['measurement'][0, 0][1][0, i][3][0][0][2][0, :]

        _len = np.min([gated.shape[0], id.shape[0], vds.shape[0]])
        # https://ieeexplore.ieee.org/document/8764332/authors#authors
        mask = gated > 2*gated.mean()
        mask = mask[:_len]
        if mask.any():
            vds = vds[:mask.shape[0]]
            id = id[:mask.shape[0]]
            rds_on = vds[mask] / id[mask]
            rds_on = rds_on.mean()
            rds.append(rds_on)


            t = X['measurement'][0, 0][1][0][i][0][0]
            time.append(int(parser.parse(t).timestamp()))
            raw_time.append(t)
            idl.append(id.mean())
            vdsl.append(vds.mean())
            vsignall.append(vsignal.mean())
            vsourcel.append(vsource.mean())




    data2 = pd.DataFrame({'time': time, 'rds': rds, 'id': idl, 'vds': vdsl, 'vsignal': vsignall, 'vsource': vsourcel})
    data2 = data2.groupby('time').mean().reset_index()
    data2.set_index('time', inplace=True)

    data2 = data2.iloc[7:]

    data3 = data.groupby('time').mean().reset_index()
    data3.set_index('time', inplace=True)


    data = data3.join(data2, how='inner').reset_index()

    X = data
    X.columns = ['time', 'sV', 'pT', 'ssVs',
       'ssId', 'fT', 'rds', 'id', 'vds', 'gvsignal',
       'gvsource']
    _mean = X.rds.values[-int(0.9*X.shape[0]):].mean()
    _std = X.rds.values[-int(0.9 * X.shape[0]):].std()

    X = X[(X.rds >= (_mean - 2*_std)) &  (X.rds <= (_mean + 2*_std))].copy()


    X['device'] = device
    X['run'] = run




    return X

VALID_DEVICES = [8,9,11,12,13,14,18,19,20,23,24,25,33,35,36,37,42]
RDSON_THRESOLD = 0.05
#RDSON_THRESOLD = 0.0

def read_data(file_path, task: dict = None, filters: dict = None):

    def ffilter(dirs, files):
        files = [f for f in files if int(os.path.split(f)[1].split('_')[1]) in VALID_DEVICES]

        return dirs, files

    D = b.read_dataset(file_path, None, ffilter=ffilter, fread_file=read_file, fprocess=None,
                          show_pbar=True, data_file_deep=2)

    D = D[0]
    #X['time'] = range(X.shape[0])

    aux = []
    for device in D.device.unique():
        X = D[D.device == device].copy()
        X['rds_norm'] = X.rds / X.fT
        X['minute'] = (X.time / 60).astype(int)
        X = X.groupby('minute').mean()
        X['rds_norm'] = X['rds_norm'] - X['rds_norm'].values[0]

        _mean, _std = X.rds_norm.mean(), X.rds_norm.std()
        _norm = (_mean + 2*_std)
        X['rds_norm'] = X.rds_norm.clip(-_norm, _norm)
        X['rds_norm'] = X['rds_norm'] - X['rds_norm'].values[0]
        X['rds_norm'] = X.rds_norm.values * 400

        no_valid =  np.where(X.rds_norm.values > 1)[0]

        if len(no_valid) > 0:
            minute = X.index[no_valid[0]]
            X = X[X.index < minute]

        X.reset_index(inplace=True)
        X['minute'] = range(0, X.shape[0])
        fail_minute = np.where(X.rds_norm.values > RDSON_THRESOLD)[0][0]
        X['rul'] = fail_minute - X.minute
        X['rul'] = X.rul.clip(0, X.rul.max())
        aux.append(X)

        #generate_figures(device, X)


        del X['minute']
        del X['time']
        del X['rds']
        del X['rds_norm']
        del X['run']

    X = pd.concat(aux, axis=0)

    #generate_figures(X.device.loc[0], X)

    return X
