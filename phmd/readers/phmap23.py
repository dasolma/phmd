import os
from phmd.readers import base as b
import pandas as pd


FAULTS = ["normal", "sv1_0", "sv1_25", "sv1_50", "sv1_75", "sv2_0", "sv2_25", "sv2_50", "sv2_75", "sv3_0", "sv3_25", 
          "sv3_50", "sv3_75", "sv4_0", "sv4_25", "sv4_50", "sv4_75", "bp1", "bp2", "bp3", "bp4", "bp5", "bp6", "bp7",
          "bv1"]


def get_fault(s):
    if s.Condition == 'Normal':
        return 0
    elif s.SV1 != 100:
        return FAULTS.index('sv1_' + str(s.SV1))
    elif s.SV2 != 100:
        return FAULTS.index('sv2_' + str(s.SV2))
    elif s.SV3 != 100:
        return FAULTS.index('sv3_' + str(s.SV3))
    elif s.SV4 != 100:
        return FAULTS.index('sv4_' + str(s.SV4))
    else:
        bool_cols = ['BP1', 'BP2', 'BP3', 'BP4', 'BP5', 'BP6', 'BP7', 'BV1']
        col = [c for c in bool_cols if s[c] == 'Yes'][0]
        return FAULTS.index(col.lower())



def read_labels(f, file_path, _):
    X = pd.read_excel(f, sheet_name=0)
    
    X.columns = list(X.columns[:3]) + list(X.iloc[0].values[3:])
    X = X.iloc[1:]
    
    X['fault'] = X.apply(get_fault, axis=1)
    X = X[['Case#', 'Spacecraft#', 'fault']]

    return X

def read_file(f, file_path, _, labels):
    case = float(os.path.split(file_path)[1].split("_")[-1].replace('.csv', '').lower().replace('case', ''))

    X = pd.read_csv(f)

    spacecraft = labels[labels['Case#'] == case]['Spacecraft#'].iloc[0]
    X['unit'] = spacecraft + (case / 100)
    X['fault'] = labels[labels['Case#'] == case].fault.iloc[0]
    X['spacecraft'] = spacecraft

    del X['TIME']

    return X


def read_data(file_path, task: dict = None, filters: dict = None):


    def ffilter(dirs, files):
        files =[f for f in files if 'labels' in f]
        return dirs, files

    L = b.read_dataset(file_path, None, ffilter=ffilter, fread_file=read_labels, fprocess=None,
                          show_pbar=True, data_file_deep=3)
    L = L[0]

    def ffilter(dirs, files):
        files =[f for f in files if 'labels' not in f]
        return dirs, files


    def _read_file(f, file_path, _):
        return read_file(f, file_path, _, L)

    X = b.read_dataset(file_path, None, ffilter=ffilter, fread_file=_read_file, fprocess=None,
                          show_pbar=True, data_file_deep=3)

    X = X[0]
    return X
