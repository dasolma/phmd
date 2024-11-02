from phmd.readers import base as b
import pandas as pd
import numpy as np

FAULTS = ["no fault", "Dampers are closed during heating regime", "Fans are OFF during heating regime",
          "Heating valve is closed during heating regime",
          "Heating valve is open to the maximum level during heating regime",
          "Heating pump is OFF during heating regime",
          "Heating valve is stuck in intermediate position during heating regime",
          "Zone inlet temperature sensor indicates -20 °C", "Zone outlet temperature sensor indicates 150 °C ",
          "Zone outlet temperature sensor indicates -20 ° C", "Heat exchanger is closed",
          "Cooling pump is ON during heating regime", "Quick regimes cycling",
          "One of fan differential pressure sensor tubes is disconnected",
          "Both fan differential pressure sensor tubes are disconnected", "Dampres are closed during ventilate regime",
          "Heating pump is ON during ventilate regime", "Cooling valve is closed during cooling regime",
          "Heating pump is ON and valve is opened during ventilate regime",
          "Heating valve is ON during ventilate regime", "Dampers are closed during cooling regime",
          "Zone inlet temperature sensor indicates 150 °C",
          "Heating valve is stuck in intermediate position during cooling regime",
          "Heating valve is open to the maximum level during cooling regime",
          "Heating valve is OFF during humidifying regime", "Heating pump is OFF during humidifying regime"]


VALID_FAULTS = ["no fault", "Dampers are closed during heating regime", "Heating valve is closed during heating regime", "Heating valve is open to the maximum level during heating regime", "Heating pump is OFF during heating regime", "Heating valve is stuck in intermediate position during heating regime", "Zone outlet temperature sensor indicates 150 °C ", "Zone outlet temperature sensor indicates -20 ° C", "Heat exchanger is closed", "Cooling pump is ON during heating regime", "One of fan differential pressure sensor tubes is disconnected", "Dampres are closed during ventilate regime", "Heating pump is ON during ventilate regime", "Cooling valve is closed during cooling regime", "Heating valve is ON during ventilate regime", "Dampers are closed during cooling regime"]


def read_file(f, file_path, _):
    X = pd.read_excel(f)
    X['fault_type'] = X.fault.map(lambda x: -1 if x not in VALID_FAULTS else VALID_FAULTS.index(x))
    X['fault'] = (X.fault_type != 0).astype('int')

    del X['Time']

    X = X.ffill()

    sample_cuts = [-1] + list(np.where(X.fault.values[1:] != X.fault.values[:-1])[0]) + [X.shape[0]]

    X['unit'] = np.nan
    for i, (l, r) in enumerate(zip(sample_cuts[:-1], sample_cuts[1:])):
        X.loc[l + 1:r, 'unit'] = X['equipment'] + "_" + str(i + 1)


    X = X[X.fault_type >= 0]

    return X


def read_data(file_path, task: dict = None, filters: dict = None):
    def ffilter(dirs, files):
        # files = [f for f in files if 'AHU8' in f]
        return dirs, files

    X = b.read_dataset(file_path, None, ffilter=ffilter, fread_file=read_file, fprocess=None,
                       show_pbar=True, data_file_deep=2)
    X = X[0]

    cols = (~X.isnull().all())
    X = X[cols.index[cols]]

    na_replacement = X[X.columns[X.isnull().any()]].min().to_dict()
    for col, v in na_replacement.items():
        X[col].fillna(v, inplace=True)
    
    fault_count = X.groupby('fault').unit.unique().map(lambda x: len(x))
    valid_faults = fault_count[fault_count >= 3].index

    X = X[X.fault.isin(valid_faults)]

    units_length = X.groupby('unit').size()
    valid_units = units_length[units_length >= 20].index

    X = X[X.unit.isin(valid_units)]

    if task['target'] == 'fault_type':
        X = X[X.fault > 0]
        X['fault_type'] = X.fault_type - 1


    return X
