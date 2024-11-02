import logging
import pandas as pd
import os 
import zipfile
import tqdm
from collections.abc import Iterable
import numpy as np
import gc
import tempfile

BIN_EXTENSSIONS = ['.h5', '.mat', '.xlsx', '.pk']


def lin_interpolate_col(X, col, by):

    for id in X[by].unique():
        values = X.loc[X[by] == id, col].values
        if np.isnan(values[0]):
            values[0] = next(v for v in values if not np.isnan(v))

        xvals = np.arange(0, values.shape[0])
        x, y = zip(*[(x, y) for x, y in zip(xvals, values) if not np.isnan(y)])

        X.loc[X[by] == id, col] = np.interp(xvals, x, y)

    return X

def ensure_iterable(obj):
    if not isinstance(obj, list) and not isinstance(obj, tuple):
        return [obj]
    else:
        return obj

def in_filename_filter(query, files):

    query = ensure_iterable(query)
    return list(filter(lambda x: any(str(s) in x for s in query), files))


def read_files(file_path, dirs, files, z=None, ffilter = None, fread_file=None, fprocess=None, join=None,
               show_pbar=True, sort_files=True, concat_delegated=False, concat_in_file=False):
    if ffilter is not None:
        dirs, files = ffilter(dirs, files)
        
    datasets = []
    concat_delegated = concat_delegated or concat_in_file
    if concat_delegated:
        datasets.append(None)

    with tqdm.tqdm(total=len(files), desc="Reading dataset", disable=(not show_pbar)) as pbar:
        concat_fp = None
        total_rows = 0
        _types = None
        col_names = None
        df_dtypes = None
        if concat_in_file:
            concat_fp = tempfile.NamedTemporaryFile("wb", delete=False)

        for data in dirs:
            data_name = data.split('/')[-1].split('.')[0]
            pbar.set_description("Reading %s" % data_name)
            #TODO: not valid in all systems "/"
            data_files = [f for f in files if data + '/'  in f or f.startswith(data)]
            if sort_files:
                data_files = sorted(data_files)

            ds = []
            if concat_delegated:
                ds.append(None)

            for i, data_file in enumerate(data_files):
                ext = os.path.splitext(data_file)[1]
                mode = 'rb' if ext in BIN_EXTENSSIONS else 'r'
                if z is None:
                    
                    with open(os.path.join(file_path, data_file), mode) as f:
                        if concat_delegated:
                            if concat_in_file:
                                X = fread_file(f, data_file, data, Xp=concat_fp)

                                col_names = list(X.columns)
                                m, n = X.shape
                                total_rows += m
                                concat_fp.write(X.values.tobytes())
                                df_dtypes = X.dtypes
                                _types = X.values.dtype
                            else:
                                ds[0] = fread_file(f, data_file, data, Xp=ds[0])
                        else:
                            ds.append(fread_file(f, data_file, data))

                else:
                    with z.open(data_file) as f:
                        if concat_delegated:
                            if concat_in_file:
                                X = fread_file(f, data_file, data, Xp=concat_fp)

                                col_names = list(X.columns)
                                m, n = X.shape
                                total_rows += m
                                concat_fp.write(X.values.tobytes())
                                df_dtypes = X.dtypes
                                _types = X.values.dtype
                            else:
                                ds[0] = fread_file(f, data_file, data, Xp=ds[0])
                        else:
                            ds.append(fread_file(f, data_file, data))
                
                pbar.update()


            if (not concat_fp):

                if (join is not None):
                    ds = join(ds)

                if len(ds) > 0:
                    X = [ensure_iterable(x) for x in ds]
                    X = [ds[0] if len(ds) == 1 else pd.concat(ds, axis=0).reset_index(drop=True) for ds in zip(*X)]

                    if fprocess is not None:
                        X = fprocess(X)

                    if len(X) == 1:
                        X = X[0]

                    if concat_delegated:
                        datasets[0] = pd.concat((datasets[0], X), ignore_index=True)
                    else:
                        datasets.append(X)

    if concat_fp:
        del ds
        gc.collect()
        concat_fp.close()
        #with open(concat_fp.name, 'rb') as f:
            #buffer = f.read()
        data = np.fromfile(concat_fp.name, dtype=_types)
        data = data.reshape(total_rows, n)
        datasets = [pd.DataFrame(data=data, columns=col_names)]
        os.remove(concat_fp.name)


    return datasets

def read_dataset(file_path, task:dict = None, ffilter = None, fread_file=None, fprocess=None, join=None, show_pbar=True,
                 data_file_deep=3, sort_files=True, concat_delegated=False, concat_in_file=False):

    if not os.path.exists(file_path):
        raise Exception('Dataset not found')

    if os.path.isdir(file_path):
        dirs = [d for d in sorted(os.listdir(file_path)) if os.path.isdir(d)]
        if len(dirs) == 0:
            dirs = [file_path]
        files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(file_path) for f in filenames]
        datasets = read_files(file_path, dirs, files, ffilter=ffilter, fread_file=fread_file,
                              fprocess=fprocess, join=join, show_pbar=show_pbar,
                              sort_files=sort_files, concat_delegated=concat_delegated,
                              concat_in_file=concat_in_file)

    else:
        with zipfile.ZipFile(file_path) as z:
            ilen = lambda x: len(list(filter(None, x.split('/'))))
            isdir = lambda name: any(x.startswith("%s/" % name.rstrip("/")) for x in z.namelist())
            files = [f for f in z.namelist() if ilen(f) == data_file_deep and not isdir(f)]
            dirs = sorted(set(['/'.join(f.split('/')[:-1]) for f in files]))
            datasets = read_files(file_path, dirs, files, z=z, ffilter=ffilter, fread_file=fread_file,
                                  fprocess=fprocess, join=join, show_pbar=show_pbar,
                                  concat_delegated=concat_delegated, concat_in_file=concat_in_file)

    if len(datasets) == 0:
        print("Error: something were wrong. No data found.")
        return None
    elif len(datasets) == 1:
        X = datasets
    elif len(datasets) > 1:
        X = [ensure_iterable(x) for x in datasets]
        logging.info("Concatenating subsets")

        X = [ds[0] if len(ds) == 1 else concat(ds) for ds in zip(*X)]

    if task is not None and len(X) == 1:
        X = X[0]
        columns = list(set(task['identifier'] + task['features'] + [task['target']]))
        return X[columns]
    else:
        return X


def concat(ds):
    return pd.concat(ds, axis=0).reset_index(drop=True)


