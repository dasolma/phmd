import pandas as pd
import os
import tqdm
from phmd.readers import base as b
import numpy as np

AColumns = ['component', 'time', 's1', 's2', 's3', 's4', 'r1', 'r2', 'r3', 'r4']
BColumns = ['zone', 'time', 'e1', 'e2']
CColumns = ['start_time', 'end_time', 'fault']

FAULTS = {
    'NOCOMMS': 6,
    'DATACHECK': 6,
    'DeltaT1CoolingZoneInSpec': 1,
    'HeatingZoneNotSpec': 6,
    'strategies.xml': 6,
    'DeltaT1CoolingZoneNotSpec': 6,
    'SupplyTempTooCold': 5,
    'RuntimeExcessive': 6,
    'scheduler.xml': 6,
    'CoolingZoneNotSpec': 4,
    'OSLReadingsLow': 6,
    'ShortCircuit': 6,
    'DeltaT2CoolingZoneNotSpec': 3,
    'OutputSetToOff': 6,
    'OutputSetToOn': 6,
    'SaveSiteSchedule': 6,
    'modules.xml': 6,
    'NoPowerLightingContactor': 6,
    'DeltaT2HeatingZoneInSpec': 6,
    'DeltaT1HeatingZoneInSpec': 6,
    'StrategyOff': 6,
    'Overrides.xml': 6,
    'FanOnlyDeltaTPositive': 6,
    'HighCO2': 6,
    'DeltaT2CoolingZoneInSpec': 2,
    'ZoneOutOfSpecNoAction': 6,
    'DeltaT1HeatingZoneNotSpec': 6,
    'ZoneOutOfSpec': 6,
    'HvacSupplyBelowFreezing': 6,
    'psychrometrics.xml': 6,
    'dcv-config.xml': 6,
    'demand-config.xml': 6,
    'PossibleOvercurrent': 6,
    'FanOnlyDeltaTNegative': 6,
    'SaveSiteInfo': 6,
    'HvacSupplyNegative': 6,
    'OutputNotChanged': 6,
    'DeltaT2HeatingZoneNotSpec': 6,
    'HvacSupplyHigh': 6,
    'HvacZoneNearFreezing': 6,
    'PowerFactor': 6,
    'KwhNotAdvancing': 6,
    'HvacZoneNegative': 6,
    'SaveSiteOverrides': 6,
    'OpenCircuit': 6,
    'HvacZoneNearBoiling': 6,
    'OSLReadingsHigh': 6,
    'OSLSensorError': 6,
    'OSLLightsStillOn': 6
}

COMPONENTS = ['ZONE', 'HVAC1', 'HVAC5', 'HVAC6', 'HVAC2', 'HVAC3', 'HVAC4', 'HVAC8',
              'HVAC7', 'HVAC9', 'HVAC10', 'HVAC11', 'HVAC12', 'HVAC13', 'HVAC14',
              'HVAC15', 'HVAC19', 'HVAC20', 'HVAC18', 'HVAC16', 'HVAC17']

ZONES = ['OTHER', 'DEMAND1', 'DEMAND2', 'METER', 'DEMAND3', 'DEMAND4',
         'DEMAND5']

R4 = ['Other', 'Occupied', 'Setback', 'OFF']

SITES = []

def read_file(f, file_path, file_name, B=None, Xp=None):
    if 'a.csv' in file_path:
        cols = AColumns
        site = os.path.split(file_path)[1].replace('a.csv', '')
    elif 'b.csv' in file_path:
        cols = BColumns
        site = os.path.split(file_path)[1].replace('b.csv', '')
    else:
        cols = CColumns
        site = os.path.split(file_path)[1].replace('c.csv', '')

    X = pd.read_csv(f, names=cols)
    X['site'] = int(site.replace('site', ''))

    if 'zone' in X.columns:
        X['zone'] = X.zone.map(lambda c: ZONES.index(c)).astype('int')
        X['time'] = (pd.to_datetime(X.time) - pd.to_datetime('1900/01/01')).dt.total_seconds()

    if 'component' in X.columns:
        X['component'] = X.component.map(lambda c: COMPONENTS.index(c))
        X = X[~X.r1.isnull()]
        X['time'] = (pd.to_datetime(X.time) - pd.to_datetime('1900/01/01')).dt.total_seconds()
        X['r4'] = X.r4.map(lambda c: R4.index(c))

    if 'fault' in X.columns:
        X['start_time'] = (pd.to_datetime(X.start_time) - pd.to_datetime('1900/01/01')).dt.total_seconds()
        X['end_time'] = X.end_time.map(lambda x: np.nan if x == '\\N' else x)
        X['end_time'] = (pd.to_datetime(X.end_time) - pd.to_datetime('1900/01/01')).dt.total_seconds()
        X['fault'] = X.fault.map(lambda f: FAULTS[f])

    if B is not None:
        X = pd.merge(X, B[B.site == site], on=['site', 'time'], how='outer')
        X.component.fillna(0, inplace=True)
        X.zone.fillna(0, inplace=True)
        X.r4.fillna(0, inplace=True)
        X.fillna(-9999, inplace=True)

    return X


def read_data(file_path, task: dict = None, filters: dict = None):
    def ffilter(dirs, files):
        files = [f for f in files if 'b.csv' in f]
        return dirs, files

    B = b.read_dataset(file_path, None, ffilter=ffilter, fread_file=read_file, fprocess=None,
                       show_pbar=True, data_file_deep=3, concat_in_file=True)
    B = B[0]

    def ffilter(dirs, files):
        files = [f for f in files if 'a.csv' in f]
        return dirs, files

    def read_file_proxy(f, file_path, file_name, Xp=None):
        return read_file(f, file_path, file_name, B)

    X = b.read_dataset(file_path, None, ffilter=ffilter, fread_file=read_file_proxy, fprocess=None,
                       show_pbar=True, data_file_deep=3, concat_in_file=True)
    X = X[0]

    def ffilter(dirs, files):
        files = [f for f in files if 'c.csv' in f]
        return dirs, files

    C = b.read_dataset(file_path, None, ffilter=ffilter, fread_file=read_file, fprocess=None,
                       show_pbar=True, data_file_deep=3, concat_in_file=True)

    C = C[0]
    C = C[C.fault != 6]

    X = X.sort_values(['site', 'time'])
    X = X.reset_index(drop=True)
    # X = pd.merge(X, C, on=['site', 'time'], how='outer')

    C = C[~C.end_time.isnull()]
    C = C.sort_values(['site', 'start_time', 'end_time'])
    C = C.values

    Xcols = list(X.columns)
    Xaux = X.values

    current_site = None
    cdata = []
    i, j = 0, 0
    with tqdm.tqdm(total=Xaux.shape[0], desc='Computing target') as pbar:
        while i < Xaux.shape[0] and j < C.shape[0]:
            rx = Xaux[i, :]
            rc = C[j, :]

            fstart, fend = rc[0], rc[1]
            t = rx[Xcols.index('time')]
            site = rc[-1]

            if (fend < t and (site == rx[Xcols.index('site')])) or (site < rx[Xcols.index('site')]):
                j += 1

            elif fstart <= t and fend >= t:
                cdata.append(rc[-2])
                i += 1

                # assert site == rx[Xcols.index('site')]

                pbar.update(1)
            else:
                cdata.append(np.nan)
                i += 1

                pbar.update(1)
    
    X.loc[0:len(cdata)-1, 'fail'] = cdata
    X = X[~X.isnull().any(axis=1)]
    del X['time']

    return X
