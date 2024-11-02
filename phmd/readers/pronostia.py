import pandas as pd
import os
import tqdm

HEADERS = ['Hour', 'Minute', 'Second', 'Microsecond', 'H_acc', 'V_acc']

TEST_RULS = {
    'Bearing1_3': 5730,
    'Bearing1_4': 339,
    'Bearing1_5': 1610,
    'Bearing1_6': 1460,
    'Bearing1_7': 7570,
    'Bearing2_3': 7530,
    'Bearing2_4': 1390,
    'Bearing2_5': 3090,
    'Bearing2_6': 1290,
    'Bearing2_7': 580,
    'Bearing3_3': 820
}

def read_file(f, file_path, bearing):
    X_aux = pd.read_csv(f, names=HEADERS, delimiter=",")
    bearing = bearing.split('/')[-1] 
    condition, number = int(bearing[7]), int(bearing[9])
    #X_aux['Bearing'] = number
    #X_aux['Bearing'] = X_aux.Bearing.astype('int8')
    #X_aux['condition'] = condition - 1
    #X_aux['condition'] = X_aux.condition.astype('int8')
    X_aux['unit'] = f"{condition}_{number}"

    del X_aux['Hour']
    del X_aux['Minute']
    del X_aux['Second']
    del X_aux['Microsecond']
    
    return X_aux


def filter_files(dirs, files, filters:dict = None):
    filtered_files = []
    filtered_dirs = []
    for bearing in dirs:
        bearing_name = bearing.split('/')[-1]
        condition, number = bearing_name[7], bearing_name[9]

        if filters is not None:
            if 'Bearing' in filters and number not in filters['Bearing']:
                continue

            if 'condition' in filters and condition not in filters['condition']:
                continue

        filtered_dirs.append(bearing)
        bearing_files = sorted([f for f in files if bearing in f and 'acc' in f])
        filtered_files += bearing_files

    return filtered_dirs, filtered_files

def read_files(file_path, dirs, files, z=None, RULS=None, filters:dict = None):

    dirs, files = filter_files(dirs, files, filters)
    datasets = []

    with tqdm.tqdm(total=len(files), desc="Reading dataset") as pbar:

        for bearing in dirs:
            bearing_name = bearing.split('/')[-1]
            pbar.set_description("Reading %s" % bearing_name)
            bearing_files = sorted([f for f in files if bearing in f and 'acc' in f])

            ds = []
            for i, bearing_file in enumerate(bearing_files):
                if z is None:
                    with open(os.path.join(file_path, bearing_file), 'r') as f:
                        ds.append(read_file(f, bearing_file, bearing))

                else:
                    with z.open(bearing_file) as f:
                        ds.append(read_file(f, bearing_file, bearing))
                
                pbar.update()


            X = pd.concat(ds, axis=0)
            X = X.reset_index(drop=True)

            # compute RUL
            X['RUL'] = (X.index / 256).astype('int')[::-1].values
            
            if RULS is not None:
                X['RUL'] += RULS[bearing.split('/')[-1]]
                
            X.RUL = X.RUL.astype('int32')
            datasets.append(X)

    return datasets

def read_dataset(file_path, RULS=None, task:dict = None, filters:dict = None):    
    datasets = []

    if os.path.isdir(file_path):
        dirs = sorted(os.listdir(file_path))
        files = [os.path.join(_dir, file) for _dir in dirs for file in os.listdir(os.path.join(file_path, _dir))]
        datasets = read_files(file_path, dirs, files, RULS=RULS, filters=filters)

    else:
        with zipfile.ZipFile(file_path) as z:
            ilen = lambda x: len(list(filter(None, x.split('/')))) 
            files = [f for f in z.namelist() if  ilen(f) == 4]
            dirs = sorted(set(['/'.join(f.split('/')[:-1]) for f in files]))
            datasets = read_files(file_path, dirs, files, z=z, RULS=RULS, filters=filters)
    
    X = pd.concat(datasets, axis=0)
    X['H_acc'] = X['H_acc'].astype('float32')
    X['V_acc'] = X['V_acc'].astype('float32')
    X['RUL'] = X['RUL'].astype('int32')
    X.rename({'RUL': 'rul'}, axis="columns", inplace=True)

    if task is not None:
        columns = list(set(task['identifier'] + task['features'] + [task['target']]))

        return X[columns]
    else:
        return X

def read_test(file_path, task:dict = None, filters:dict = None):
    return read_dataset(file_path, RULS=TEST_RULS, task=task, filters=filters)