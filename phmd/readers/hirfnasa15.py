import pandas as pd
import os
import scipy as scipy
from phmd.readers import base as b

def read_file(f, file_path, _, Xp=None):

    unit = int(os.path.split(file_path)[1].replace('.mat', '').replace('HIRF', ''))
    data = scipy.io.loadmat(f)


    data_cols = [t[0] for t in eval(str(data['data'][0, 0].dtype))]

    if data_cols[-1] == 'BHM':
        estimate = data['data'][0, 0][-1][0,0]
        data_cols = data_cols[:-1]
    elif 'estimate' in data.keys():
        estimate = data['estimate'][0, 0]
    else:
        return pd.DataFrame()

    data_dict = {}
    for i in range(len(data_cols)):
        data_dict[data_cols[i]] = data['data'][0, 0][i][:,0]

    D = pd.DataFrame(data_dict)

    estimate_cols = [t[0] for t in eval(str(estimate.dtype))]
    estimate_dict = {}
    for i in range(len(estimate_cols)):
        d = estimate[i]
        if d.shape[1] > 1:
            estimate_dict[estimate_cols[i]] = d.mean(axis=1)
        else:
            estimate_dict[estimate_cols[i]] = d[:, 0]

    E = pd.DataFrame(estimate_dict)

    '''
    if 'prognosis' in data.keys():
        prognosis = data['prognosis'][0, 0]
        prognosis_cols = [t[0] for t in eval(str(prognosis.dtype))]
        prognosis_dict = {}
        for i in range(len(prognosis_cols)):
            d = prognosis[i]
            if d.shape[1] > 1:
                prognosis_dict[prognosis_cols[i]] = d.mean(axis=1)
            else:
                prognosis_dict[prognosis_cols[i]] = d[:, 0]

        P = pd.DataFrame(prognosis_dict)
    '''

    D = pd.merge(D, E, how='left').ffill()
    D = D[~D.isnull().any(axis=1)]

    experiments = ['LLF', 'LRF', 'ULA', 'URA']

    compose_columns = lambda exp: [c for c in D.columns if c in (['t', 'RPM'] + [exp + c for c in ['20V', '20C', '20T', '40V', '40C', '40T', '_SOC']])]
    def ren(D, exprs):
        if len(exprs) == 0:
            return D.copy()
        else:
            cols = D.columns
            if any(exprs[0] in c for c in cols):
                D = D.rename({c:c.replace(exprs[0], '').lower() for c in D.columns}, axis='columns')
                D['unit'] = exprs[0]
                return D.copy()
            else:
                return ren(D, exprs[1:])


    DS = [ren(D[compose_columns(e)], experiments) for e in experiments]
    D = pd.concat(DS, axis=0)
    D['fail'] = D._soc < 40

    TF = D[D.fail].groupby('unit').t.min().reset_index()
    TF.columns = ['unit', 'fail_t']

    D = pd.merge(D, TF)
    D['rul'] = (D.fail_t - D.t).clip(0, 1000000)
    D['unit'] = D['unit'].map(lambda x: unit + experiments.index(x) / 10)
    D = D.rename({'_soc': 'soc'}, axis='columns')

    del D['t']
    del D['fail']
    del D['fail_t']

    D = D.fillna(-100)

    return D


def read_data(file_path, task: dict = None, filters: dict = None):

    def ffilter(dirs, files):
        return dirs, files

    X = b.read_dataset(file_path, None, ffilter=ffilter, fread_file=read_file, fprocess=None,
                          show_pbar=True, data_file_deep=3,  concat_delegated=True, concat_in_file=True)
    X = X[0]
    _max = X.groupby('unit').rul.transform('max')
    _index = X.index[_max == 0]

    X.drop(index=_index, inplace=True)

    return X
