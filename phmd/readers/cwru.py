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


OUTER_FAULTS = {
    "Centered": 6,
    "Orthogonal": 3,
    "Opposite": 12
}

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
        key_map = {k: get_mat_key(k) for k in ['DE', 'FE']}

        nsamples = struct[list(key_map.values())[0]].shape[0]
        data = {k: struct[key_map[k]].reshape(nsamples, )
                for k in key_map.keys()
                if key_map[k] is not None}

        data['unit'] = _id
        data['fault'] = fault

        return pd.DataFrame(data)

    assert len(sample_ids) > 0

    X = pd.concat([extract_sample(_id) for _id in sample_ids])

    #assert ~X.isnull().any().any()

    return X


def read_data(file_path, task: dict = None, filters: dict = None):
    def __read_file(f, file_path, _):
        return read_file(f, file_path, filters)

    def ffilter(dirs, files):
        if filters is not None: # Si hay filtros disponibles
            if 'files' in filters: # Filtro de nombres de archivo
                # Filtrar la lista files según los nombres especificados en filters['files']:
                files = in_filename_filter(filters['files'], files)

            if 'speed' in filters: # Filtro de nombres de archivo según la velocidad
                files = in_filename_filter(filters['speed'], files)

            if 'fault_diameter' in filters: # Filtro de diámetro de falla
                normal_files = []
                diameters = filters['fault_diameter'] # Diámetros de falla especificados
                if 0 in filters['fault_diameter']: # Archivos sin fallas
                    normal_files = [f for f in files if 'Normal' in f]
                # Se filtran los archivos según los diámetros especificados en filters['fault_diameter']:
                diameters = [d for d in diameters if str(d) != '0']
                files = in_filename_filter(diameters, files)
                # Se concatenan con los archivos sin falla:
                files += normal_files

            if 'sample_rate' in filters: # Filtro de tasa de muestreo
                # Lista de cadenas que representan las tasas de muestreo especificadas en filters['sample_rate']:
                sample_rates = ['/' + str(sr) for sr in filters['sample_rate']]
                if '/48' in sample_rates: # Si '/48' está en sample_rates
                    sample_rates.append('/Normal') # Se añade '/Normal' a sample_rates

                # Se filtran los archivos según las tasas de muestreo especificadas en sample_rates:
                files = in_filename_filter(sample_rates, files)

            if 'outer_race_faults' in filters: # Filtro de fallas en la carrera externa
                # Lista con las categorías numéricas de fallas externas:
                outer_faults = [str(OUTER_FAULTS[of]) for of in filters['outer_race_faults']]
                # Se filtran los archivos que contienen 'OuterRace' pero cuyo último o penúltimo carácter del nombre
                # del archivo coincide con outer_faults
                files = [f for f in files
                         if (('OuterRace' not in f) or
                             (f.replace('.mat', '')[-1] in outer_faults) or
                             (f.replace('.mat', '')[-2] in outer_faults))]

        return dirs, files

    Xs = b.read_dataset(file_path, None, ffilter=ffilter, fread_file=__read_file, fprocess=None,
                       show_pbar=True, data_file_deep=5)

    X = Xs[0]
    X = X[~X.isnull().any(axis=1)]

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
