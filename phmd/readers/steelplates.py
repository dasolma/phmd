import pandas as pd
from phmd.readers import base as b
import numpy as np

HEADERS = ["X_Minimum", "X_Maximum", "Y_Minimum", "Y_Maximum", "Pixels_Areas", "X_Perimeter", "Y_Perimeter",
           "Sum_of_Luminosity", "Minimum_of_Luminosity", "Maximum_of_Luminosity", "Length_of_Conveyer",
           "TypeOfSteel_A300", "TypeOfSteel_A400", "Steel_Plate_Thickness", "Edges_Index", "Empty_Index",
           "Square_Index", "Outside_X_Index", "Edges_X_Index", "Edges_Y_Index", "Outside_Global_Index", "LogOfAreas",
           "Log_X_Index", "Log_Y_Index", "Orientation_Index", "Luminosity_Index", "SigmoidOfAreas", "Pastry",
           "Z_Scratch", "K_Scatch", "Stains", "Dirtiness", "Bumps", "Other_Faults"]

FAULTS = ["Pastry", "Z_Scratch", "K_Scatch", "Stains", "Dirtiness", "Bumps", "Other_Faults"]
def read_file(f, file_path, bearing):


    X = pd.read_csv(f, sep='\t', names=HEADERS)

    X['fault'] = X.apply(lambda x: list((x[FAULTS] == 1).values).index(True), axis=1)

    X.drop(FAULTS, axis=1, inplace=True)

    X['unit'] = np.arange(0, X.shape[0])

    return X



def read_data(file_path, task: dict = None, filters: dict = None):

    X = b.read_dataset(file_path, None, ffilter=None, fread_file=read_file, fprocess=None,
                          show_pbar=True)

    # compute RUL


    return X
