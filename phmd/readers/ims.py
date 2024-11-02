import pandas as pd
import os
from phmd.readers import base as b

HEADERS1 = ["acc1_1", "acc1_2", "acc2_1", "acc2_2", "acc3_1", "acc3_2", "acc4_1", "acc4_2"]
HEADERS23 = ["acc1", "acc2", "acc3", "acc4"]

def read_file(f, file_path, bearing, Xp):
    headers = HEADERS1 if 'IMS/unit 1' in file_path else HEADERS23
    X_aux = pd.read_csv(f, names=headers, delimiter="\t")
    X_aux['unit'] = int(os.path.split(file_path)[0][-1])

    ref = dt.strptime('2003.01.01.00.00.00', '%Y.%m.%d.%H.%M.%S')
    d = os.path.split(file_path)[1]
    seconds = dt.strptime(d, '%Y.%m.%d.%H.%M.%S') - ref
    X_aux['seconds'] = seconds.total_seconds()

    if ('acc1_2' in X_aux.columns):
        cols1 = ['acc1_1', 'acc2_1', 'acc3_1', 'acc4_1', 'unit', 'seconds']
        cols2 = ['acc1_2', 'acc2_2', 'acc3_2', 'acc4_2', 'unit', 'seconds']
        
        X1 = X_aux[cols1].copy()
        X1.columns = ['acc1', 'acc2', 'acc3', 'acc4', 'unit', 'seconds']
        X1['acc'] = 1
        X2 = X_aux[cols2].copy()
        X2.columns = ['acc1', 'acc2', 'acc3', 'acc4', 'unit', 'seconds']
        X2['acc'] = 2

        X_aux = pd.concat((X1, X2), axis=0, ignore_index=True)

    else:
        X_aux['acc'] = 1


    return X_aux



def read_data(file_path, task: dict = None, filters: dict = None):

    def filter_files(dirs, files):
        if filters is not None and 'unit' in filters:
            units = ['unit ' + str(u) for u in ensure_iterable(filters['unit'])]
            files = in_filename_filter(units, files)

        return dirs, files

    Xs = b.read_dataset(file_path, None, ffilter=filter_files, fread_file=read_file, fprocess=None,
                          show_pbar=True, concat_delegated=True, concat_in_file=True)

    # compute RUL
    X = Xs[0]

    max_seconds_per_unit = X.groupby('unit')['seconds'].transform('max')
    X['rul'] = (max_seconds_per_unit - X.seconds) / 3600

    return X
