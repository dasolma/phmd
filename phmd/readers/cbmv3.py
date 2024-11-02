from collections import defaultdict, Counter
import pandas as pd
import os
import tqdm
import numpy as np
from phmd.readers import base as b
from datetime import datetime as dt
import scipy.io

from phmd.readers.base import ensure_iterable, in_filename_filter


def read_file(f, file_path, filters):
    data = np.array([np.array([float(d) for d in l.split()]) for l in f.readlines()])
    X = pd.DataFrame(data, columns=['l', 's', 'gtst', 'gts', 'cppts', 'cpptp', 'stp', 'srp', 'sts', 'srs', 'htet', 'gogs', 'ff', 'atcs', 'gcoap', 'gcoat', 'ep', 'htep', 'ttcs', 'tcs', 'prs', 'tcp', 'prp', 'ptp', 'pts', 'ptdsc_port', 'ptdsc_stbd', 'hdsc', 'gcdsc', 'gtdsc'])
    X['unit'] = 1
    return X


def read_data(file_path, task: dict = None, filters: dict = None):
    def filter_files(dirs, files):
        return dirs, files


    def __read_file(f, file_path, _):
        return read_file(f, file_path, filters)

    Xs = b.read_dataset(file_path, None, ffilter=filter_files, fread_file=__read_file, fprocess=None, join=None,
                        show_pbar=True, data_file_deep=3)


    return Xs[0]
