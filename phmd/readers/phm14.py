from collections import defaultdict, Counter
import pandas as pd
import os
import tqdm
import numpy as np
from phmd.readers import base as b
from datetime import datetime as dt
import scipy.io

from phmd.readers.base import ensure_iterable, in_filename_filter


def read_file(f, file_path, filters):
    """
    Base on https://github.com/MichaelBosello/battery-rul-estimation
    """
    X = pd.read_csv(f)

    type = os.path.split(file_path)[1].split('-')[1].replace('.csv', '').strip().lower()

    #debug_assets = ['A403193', 'A617958', 'A531932', 'A242376', 'A650572', 'A180615', 'A603228', 'A988606', 'A183204', 'A124878', 'A764985', 'A678003', 'A421652', 'A125623', 'A305613', 'A654809', 'A558889', 'A097170', 'A640947', 'A465075', 'A210324', 'A699738', 'A704888', 'A105614', 'A558858', 'A112094', 'A737807', 'A301854', 'A521377', 'A920271', 'A298902', 'A373145', 'A712075', 'A521002', 'A356393', 'A671923', 'A315112', 'A784870', 'A166185', 'A439865', 'A657758', 'A991622', 'A866898', 'A944044', 'A645265', 'A888026', 'A996817', 'A087012', 'A321141', 'A682371', 'A837484', 'A958844', 'A552524', 'A072341', 'A153912', 'A318624', 'A092462', 'A697472', 'A057233', 'A752282', 'A129627', 'A327303', 'A885442', 'A062079', 'A040943', 'A943170', 'A897883', 'A107046', 'A776175', 'A351418', 'A432811', 'A760193', 'A658977', 'A356779', 'A115725', 'A998861', 'A399220', 'A457170', 'A966262', 'A728299', 'A976192', 'A745705', 'A674763', 'A694094', 'A340894', 'A751024', 'A291959', 'A401118', 'A000204', 'A811559', 'A891841', 'A421448', 'A121855', 'A805199', 'A502861', 'A401996', 'A556731', 'A274757', 'A991069', 'A372945']
    #X = X[X.Asset.isin(debug_assets)]

    return {type: X}


def read_data(file_path, task: dict = None, filters: dict = None):
    def filter_files(dirs, files):
        return dirs, files

    def join(Xs):
        data = {k: v for d in Xs for k, v in d.items()}

        u = data['usage']

        # interpolate usage beetween time units
        assets = {asset: u[u.Asset == asset][['Time', 'Use']].values for asset in u.Asset.unique()}
        usages = {asset: np.hstack([[np.arange(t1, t2), np.interp(np.arange(t1, t2), [t1, t2], [u1, u2])]
                                    for ((t1, u1), (t2, u2)) in
                                    zip(usages[:-1], usages[1:])]) for asset, usages in assets.items() if
                  len(usages) > 1}
        usages = [np.array([[a] * len(u[0]), u[0], u[1]]) for a, u in usages.items()]
        usages = np.hstack(usages)
        u = pd.DataFrame(usages.T, columns=['Asset', 'Time', 'Use'])
        u['Time'] = u.Time.map(lambda x: int(float(x)))
        u['Use'] = u.Use.map(lambda x: float(x))

        f = data['failures']
        f['fault'] = 1
        c = data['part consumption']

        part_count = Counter(c.Part)
        others = [k for k, v in part_count.items() if v < 100]
        c['Part'] = c.Part.map(lambda part: 'other' if part in others else part)

        cparts = c[['Asset', 'Time', 'Part', 'Quantity']].copy()
        cparts = cparts.groupby(['Asset', 'Time', 'Part']).Quantity.sum().reset_index()
        cparts = cparts.pivot(index=['Asset', 'Time'], columns='Part', values='Quantity')
        cparts.fillna(0, inplace=True)

        creason = c[['Asset', 'Time', 'Reason']].copy()
        creason['Dummy'] = 1
        creason = creason.drop_duplicates()
        creason = creason.pivot(index=['Asset', 'Time'], columns='Reason', values='Dummy')
        creason.fillna(0, inplace=True)

        c = creason.join(cparts).reset_index()

        aux = pd.merge(u, f, on=['Asset', 'Time'], how='outer').sort_values(['Asset', 'Time'])
        X = pd.merge(aux, c, on=['Asset', 'Time'], how='outer').sort_values(['Asset', 'Time'])
        X['Use'] = X.groupby('Asset').Use.ffill()

        X.fillna(0, inplace=True)

        cols = [c for c in X.columns if c.startswith('R') or c.startswith('P') or c == 'other']
        X.loc[:, cols] = X.groupby('Asset')[cols].cumsum()

        X['faults'] = X.groupby('Asset').fault.cumsum()

        # compute rul
        assets = []
        ruls = []
        times = []
        faults = []
        pt = 0
        pa = ""
        for a, t in X[X.fault == 1][['Asset', 'Time']].values:
            if pa != a:
                pt = 0

            assets += [a] * (1 + t - pt)
            r = t - np.arange(pt, t + 1)
            ruls += list(r)
            times += list(np.arange(pt, t + 1))
            faults += [v if v <= 3 else 0 for v in (r + 1)]

            pt = t
            pa = a

        R = pd.DataFrame({'Asset': assets, 'Time': times, 'rul': ruls, 'fault': faults})
        R['fault'] = R.fault.map(lambda x: 1 if x > 0 else 0)

        del X['fault']
        X = pd.merge(X, R, on=['Asset', 'Time'], how='left').sort_values(['Asset', 'Time'])
        X = X[~X.rul.isnull()]
        X = X[X.Use != 0]

        # Time is nos valid for modeling
        del X['Time']

        X.columns = [c.lower() for c in X.columns]

        return [X]

    def __read_file(f, file_path, _):
        return read_file(f, file_path, filters)

    Xs = b.read_dataset(file_path, None, ffilter=filter_files, fread_file=__read_file, fprocess=None, join=join,
                        show_pbar=True, data_file_deep=3)

    X = Xs[0]

    valid_ts_assets = (X.groupby('asset').size() > 32).reset_index()
    valid_ts_assets.columns = ['asset', 'valid']
    valid_ts_assets = valid_ts_assets[valid_ts_assets.valid].asset

    X = X[X.asset.isin(valid_ts_assets)]

    return X
