from phmd.readers import base as b
import pandas as pd
import numpy as np

FAULTS = ['FlowCool Pressure Dropped Below Limit',
          'Flowcool Pressure Too High Check Flowcool Pump',
          'Flowcool leak']

UNITS = ['01M01', '01M02', '02M01', '02M02', '03M01', '03M02', '04M01', '04M02', '05M01', '05M02', '06M01', '06M02',
         '07M01', '07M02', '08M01', '08M02', '09M01', '09M02', '10M01', '10M02']


def _data_read_file(f, file_path, F, _):
    X  = pd.read_csv(f)

    tool = X.Tool.iloc[0]
    unit = int(tool[:2]) + (int(tool[-2:])/10)
    f  = F[F.Tool == X.Tool.iloc[0]]

    del X['Tool']

    X['fault'] = np.nan
    samples = []
    for i in range(f.shape[0]):
        t = f.iloc[i].time
        idx = X.loc[X.time < t].index
        if len(idx) > 0:
            idx = idx[-1]
            mask = (X.index <= idx) & (X.index > idx-1000)
            X.loc[mask, 'fault'] = f.iloc[i].fault
            X.loc[mask, 'unit'] = unit + 100*i
            X.loc[mask, 'rul'] = np.arange(0, 1000)[::-1]
            samples.append(X.loc[mask].values)

    cols = X.columns
    X = np.concatenate(samples)
    X = pd.DataFrame(X, columns=cols)
    X.columns = cols

    X = X[~X.fault.isnull()]

    del X['time']


    #print(file_path)
    #print(X.shape)
    #print(X.isnull().any().any())
    #print(X.fault.unique())

    return X

def fault_read_file(f, file_path, _):
    X  = pd.read_csv(f)
    X = X.drop_duplicates()
    X['fault_name'] = X.fault_name.map(lambda x: x.replace(' [NoWaferID]',''))
    X['fault'] = X.fault_name.map(lambda x: FAULTS.index(x))


    return X

def read_data(file_path, task: dict = None, filters: dict = None):

    def fault_filter(dirs, files):
        files = [f for f in files if 'fault' in f]
        return dirs, files

    F = b.read_dataset(file_path, None, ffilter=fault_filter, fread_file=fault_read_file, fprocess=None,
                          show_pbar=True, data_file_deep=3)
    F = F[0]


    def data_filter(dirs, files):
        files = [f for f in files if not 'fault' in f]
        #files = files[:2]
        return dirs, files

    def data_read_file(f, file_path, _, Xp=None):
        return _data_read_file(f, file_path, F, _)

    X = b.read_dataset(file_path, None, ffilter=data_filter, fread_file=data_read_file, fprocess=None,
                          show_pbar=True, data_file_deep=3, concat_in_file=True)
    X = X[0]
    return X
