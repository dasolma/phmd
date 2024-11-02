import pandas as pd
import os
from phmd.readers import base as b

FAULTS = ["normal", "piston shoes and swashplate wearing", "valve plate wearing"]

def secure_file_read(f):
    data = ''
    ch = secure_char_read(f)
    data += ch
    while ch:
        ch = secure_char_read(f)
        data += ch

    return data.replace('#', '')

def secure_char_read(f):
    try:
        p = f.tell()
        ch = f.read(1)
    except Exception as ex:
        f.seek(p+1)
        ch = '#'


    return ch


def read_file(f, file_path, bearing):
    experiment = int(os.path.split(file_path)[1].replace(".txt", "").split('-')[1])
    fault = os.path.split(os.path.split(file_path)[0])[1]
    fault = [i for i, f in enumerate(FAULTS) if f in fault][0]

    #f.seek(146)
    data = secure_file_read(f)
    if isinstance(data, bytes):
        data = data.decode()
    data = data.strip().split("\n")
    data = [float(v) for v in data]
    if data[0] >= 1:
        data = data[1:]

    X = pd.DataFrame({'vibration': data})
    X['experiment'] = f"{experiment}_{fault}"
    X['fault'] = fault

    return X



def read_data(file_path, task: dict = None, filters: dict = None):

    def ffilter(dirs, files):
        return dirs, files

    X = b.read_dataset(file_path, None, ffilter=ffilter, fread_file=read_file, fprocess=None,
                          show_pbar=True, data_file_deep=5)

    X = X[0]
    return X
