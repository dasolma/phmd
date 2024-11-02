import os
from phmd.readers import base as b
import pandas as pd

# ["gear_norm", "gear_chipped", "gear_root", "gear_miss", "gear_surface", "bearing_normal",
                #"bearing_ball", "bearing_inner", "bearing_outer", "bearing_combination"]
FAULTS = {
    ("gearset", "Health"): 0,
    ("gearset", "Chipped"): 1,
    ("gearset", "Root"): 2,
    ("gearset", "Miss"): 3,
    ("gearset", "Surface"): 4,
    ("bearingset", "health"): 5,
    ("bearingset", "ball"): 6,
    ("bearingset", "inner"): 7,
    ("bearingset", "outer"): 8,
    ("bearingset", "comb"): 9,
}

COLUMNS = ["motor_vib", "planetary_gear_x", "planetary_gear_y", "planetary_gear_z", "motor_torque", "parallel_gear_x",
           "parallel_gear_y", "parallel_gear_z"]

def read_file(f, file_path, _):
    sep = '\t'
    if 'ball_20_0' in file_path:
        sep = ","
    left, right = os.path.split(file_path)
    dir = os.path.split(left)[1]
    fault = right.split("_")[0]
    exp = min(int(right.split("_")[2].replace('.csv', '')) + 1, 2)

    X = pd.read_csv(f, sep=sep, skiprows=16, names=COLUMNS, index_col=False)
    X["fault"] = FAULTS[(dir, fault)]
    X['unit'] = FAULTS[(dir, fault)] + (exp/10)

    return X


def read_data(file_path, task: dict = None, filters: dict = None):


    def ffilter(dirs, files):
        return dirs, files

    X = b.read_dataset(file_path, None, ffilter=ffilter, fread_file=read_file, fprocess=None,
                          show_pbar=True, data_file_deep=3)


    return X
