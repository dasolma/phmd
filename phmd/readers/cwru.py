import pandas as pd
import os
from phmd.readers import base as b
import scipy.io
from phmd.readers.base import ensure_iterable, in_filename_filter

FAULT_MAP = {
    'InnerRace': 'IR',
    'OuterRace': 'OR',
    'Ball': 'BA',
    'Normal': 'NO'
}

FAULTS = ['IR', 'OR', 'BA', 'NO']

def read_file(f, file_path, filters = {}):
    """

    """

    struct = scipy.io.loadmat(f)
    mat_keys = list(struct.keys())

    sample_ids = set([k[1: k.index('_')] for k in mat_keys if k.startswith('X') and '_' in k])
    sample_ids = [_id for _id in sample_ids if any(f'X{_id}' in k for k in mat_keys)]

    file_name = os.path.split(file_path)[1]
    fault = next(v for k, v in FAULT_MAP.items() if k in file_name)
    fault = FAULTS.index(fault)

    if filters is not None and 'fault' in filters and not fault in ensure_iterable(filters['fault']):
        sample_ids = []

    def extract_sample(_id):
        get_mat_key = lambda key: next((k for k in mat_keys if ('X%s_%s_' % (_id, key)) in k), None)
        key_map = {k: get_mat_key(k) for k in ['DE']}

        nsamples = struct[list(key_map.values())[0]].shape[0]
        data = {k: struct[key_map[k]].reshape(nsamples, )
                for k in key_map.keys()
                if key_map[k] is not None}

        data['unit'] = _id
        data['fault'] = fault

        return pd.DataFrame(data)

    assert len(sample_ids) > 0

    X = pd.concat([extract_sample(_id) for _id in sample_ids])

    assert ~X.isnull().any().any()

    return X


def read_data(file_path, task: dict = None, filters: dict = None):
    def __read_file(f, file_path, _):
        return read_file(f, file_path, filters)

    def ffilter(dirs, files):
        if filters is not None:
            if 'speed' in filters:
                files = in_filename_filter(filters['speed'], files)

            if 'fault_diameter' in filters:
                files = in_filename_filter(filters['fault_diameter'], files)

            if 'sample_rate' in filters:
                sample_rates = ['/' + str(sr) for sr in filters['sample_rate']]
                if '/48' in sample_rates:
                    sample_rates.append('/Normal')
                    files = in_filename_filter(sample_rates, files)

        return dirs, files

    Xs = b.read_dataset(file_path, None, ffilter=ffilter, fread_file=__read_file, fprocess=None,
                       show_pbar=True, data_file_deep=5)

    X = Xs[0]

    # set task columns
    if task is not None:

        target = ensure_iterable(task['target'])

        def columns(X):
            return [c for c in set(task['identifier'] + task['features'] + target) if c in X.columns]

        X = X[columns(X)]

        X.fillna(0, inplace=True)


        return X
    else:
        return X
