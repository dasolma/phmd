from collections import defaultdict
import pandas as pd
from phmd.readers import base as b
import scipy.io

from phmd.readers.base import ensure_iterable


MATERIALS = ["cast iron", "steel"]

def read_file(f, file_path, filters = {}):
    """
    Based on code: https://www.kaggle.com/vinayak123tyagi/mat-to-csv-code?scriptVersionId=35537522
    Args:
        f:
        file_path:
        filters:

    Returns:

    """
    struct = scipy.io.loadmat(f)

    # prepare filters
    units = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    materials = [1, 2]
    if filters is not None:
        units = filters.get('unit', units)

        if 'material' in filters:
            materials = [MATERIALS.index(v)+1 for v in ensure_iterable(filters['material'])]

    # parsing arrays in mat file
    COLUMNS = ['unit', 'run', 'VB', 'time', 'DOC', "feed", "material", "smcAC", "smcDC", "vib_table", "vib_spindle",
               "AE_table", "AE_spindle"]
    data = defaultdict(lambda: [])

    arr_data = struct['mill'][0]
    for i in range(len(arr_data)):
        sample = arr_data[i]
        max_len = max(len(sample[j]) for j in range(0, 13))
        if sample[0][0][0] in units and sample[6][0][0] in materials:
            for icol, col in enumerate(COLUMNS):
                values = sample[icol].tolist()
                # flatten
                values = values[0] if len(values) == 1 else [v for l in values for v in l]

                # fill with none
                values = [None] * (max_len - len(values)) + values

                data[COLUMNS[icol]] += values
        else:
            print(sample[0][0][0], sample[6][0][0])

    data = pd.DataFrame.from_dict(data)


    return data


def __add_cbm_state(X):

    def tool_state(cols):
        """Add the label to the cut. Categories are:
        Healthy Sate (label=0): 0~0.2mm flank wear
        Degredation State (label=1): 0.2~0.7mm flank wear
        Failure State (label=2): >0.7mm flank wear
        """
        # pass in the tool wear, VB, column
        vb = cols

        if vb < 0.2:
            return 0
        elif vb >= 0.2 and vb < 0.7:
            return 1
        elif pd.isnull(vb):
            pass
        else:
            return 2

    # apply the label to the dataframe
    X["CBM"] = X["VB"].apply(tool_state)

    return X



def read_data(file_path, task: dict = None, filters: dict = None):

    def __read_file(f, file_path, _):
        return read_file(f, file_path, filters)

    Xs = b.read_dataset(file_path, None, ffilter=None, fread_file=__read_file, fprocess=None,
                       show_pbar=True)

    for i in range(len(Xs)):
        X = Xs[i].copy()
        # fill unit
        X.unit.fillna(method='backfill', inplace=True)

        # interpolate VB
        X = b.lin_interpolate_col(X, 'VB', 'unit')

        # compute CMB target
        X = __add_cbm_state(X)

        if task is not None:
            # set task columns
            target = ensure_iterable(task['target'])
            columns = list(set(task['identifier'] + task['features'] + target))

            X = X[columns]

        Xs[i] = X

    X = Xs[0]
    X = X[(X.unit != 2) & (X.unit != 12)]  # bad signals

    X["unit"] = X.unit.astype('int').astype('str') + "_" + X.CBM.astype('str')
    X = X[(X.unit != "15_2") & (X.unit != "10_2")]  # bad signals
    
    return X
