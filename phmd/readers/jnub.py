import pandas as pd
import os
from phmd.readers import base as b

HEADERS = ["vibration"]

FAULTS = ["health state", "inner ring", "outer ring", "rolling element"]
FAULT_PREFIX = ["n", "ib", "ob", "tb"]

def read_file(f, file_path, bearing):
    

    X = pd.read_csv(f, sep='\t', names=HEADERS)


    bearing_name = os.path.split(file_path)[-1]

    prefix = "n"
    if bearing_name[0] != "n":
        prefix = bearing_name[:2]

    speed = int(bearing_name.replace(prefix, "").split("_")[0])
    bearing_id = 1 if bearing_name.replace(prefix, "").split("_")[0] == "2" else 2

    #X['bearing'] = bearing_id
    X['speed'] = speed
    X['fault'] = FAULT_PREFIX.index(prefix)

    X['unit'] = f"B{bearing_id}_S{speed}_F{FAULT_PREFIX.index(prefix)}"

    return X



def read_data(file_path, task: dict = None, filters: dict = None):

    def ffilter(dirs, files):
        files = [f for f in files if '.csv' in f]
        return dirs, files

    X = b.read_dataset(file_path, None, ffilter=ffilter, fread_file=read_file, fprocess=None,
                          show_pbar=True, data_file_deep=2)

    # compute RUL


    return X
