import os
import scipy as scipy
from phmd.readers import base as b
import pandas as pd


FAULTS_COMPONENT = ["EE", "IR", "OR", "RE"]
FAULTS = ["EE_F0", "OR_F1", "IR_F1", "RE_F1", "OR_F2", "IR_F2", "RE_F2",
          "OR_F3", "IR_F3", "RE_F3", "OR_F4", "IR_F4", "RE_F4",]

def read_file(f, file_path, _):
    unit, regine, freq, fault_component, fault_level, replica = os.path.split(file_path)[1].replace('.mat', '').split("_")
    data = scipy.io.loadmat(f)


    X = pd.DataFrame()
    X["Rod_1"] = data["Rod_1"][:, 0]
    X["Rod_2"] = data["Rod_2"][:, 0]
    X["Rod_3"] = data["Rod_3"][:, 0]
    X['fault_component'] = FAULTS_COMPONENT.index(fault_component)
    X['fault'] = FAULTS.index(f"{fault_component}_{fault_level}")
    X["unit"] = int(unit)

    return X


def read_data(file_path, task: dict = None, filters: dict = None):

    def ffilter(dirs, files):
        return dirs, files

    X = b.read_dataset(file_path, None, ffilter=ffilter, fread_file=read_file, fprocess=None,
                          show_pbar=True, data_file_deep=2)


    return X[0]
