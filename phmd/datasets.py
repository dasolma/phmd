import importlib
import inspect
import random
import numpy as np
from phmd import *
from phmd.download import download
import os
from phmd.readers.base import ensure_iterable
from phmd.utils import get_storage_dir, print_dict, get_hash
import json
import pandas as pd
from typing import List
import time
from sklearn.model_selection import StratifiedKFold, train_test_split, KFold
from tabulate import tabulate
import pickle as pk
import logging
from typing import List, Optional, Union, Iterator


logging.basicConfig(level=logging.INFO)

def __stratified_kfold_split_by_unit(units: List[List[object]], ifold: int, num_folds: int = 5, random_state: int = 666,
                                     targets: List[object] = None, test_pct=0.3):
    classes = np.unique(targets)
    number_units_per_target = min(list(targets).count(e) for e in np.unique(targets))
    ntest_units_per_class = max(1, int(number_units_per_target * test_pct))

    train_units, test_units, train_targets, test_targets = train_test_split(units, targets,
                                                                            test_size=ntest_units_per_class *
                                                                                      classes.shape[0],
                                                                            stratify=targets,
                                                                            random_state=random_state)

    skf = StratifiedKFold(n_splits=num_folds, random_state=random_state, shuffle=True)
    folds = list(skf.split(train_units, train_targets))

    train_idx, val_idx = folds[ifold]
    return train_units[train_idx], train_units[val_idx], test_units


def __kfold_split_by(X, by, ifold: int, num_folds: int = 5, extract_test: bool = True, random_state: int = 666,
                     test_pct=0.25):
    random.seed(random_state)

    split_cols = by
    split_assets = X[split_cols].drop_duplicates().values

    if extract_test:
        ntest_assets = max(1, int(split_assets.shape[0] * test_pct))

        test_assets = random.sample(list(split_assets), ntest_assets)
        test_mask = __extract_mask(X, split_cols, test_assets)

        # memory leak
        # test_mask = np.any(
        #    np.concatenate([X[split_cols].values == test_assets[i] for i in range(len(test_assets))], axis=1), axis=1)

        train_assets = [a for a in split_assets if a not in test_assets]
    else:
        train_assets = list(split_assets)

    skf = KFold(n_splits=num_folds, random_state=random_state, shuffle=True)
    folds = list(skf.split(train_assets))

    train_idx, val_idx = folds[ifold]
    train_assets, val_assets = np.array(train_assets)[train_idx], np.array(train_assets)[val_idx]

    train_mask = __extract_mask(X, split_cols, train_assets)
    # train_mask = np.any(
    #    np.concatenate([X[split_cols].values == train_assets[i] for i in range(len(train_assets))], axis=1), axis=1)

    val_mask = __extract_mask(X, split_cols, val_assets)
    # val_mask = np.any(
    #    np.concatenate([X[split_cols].values == val_assets[i] for i in range(len(val_assets))], axis=1), axis=1)

    if extract_test:
        return train_mask, val_mask, test_mask
    else:
        return train_mask, val_mask, None


def __extract_mask(X, split_cols, assets):
    test_mask = np.zeros((X.shape[0],)).astype('bool')
    for asset in assets:
        test_asset_mask = np.all(X[split_cols].values == asset, axis=1)
        test_mask = test_mask | test_asset_mask

    return test_mask


def __ts_split_by_unit(X, identifier, ifold: int, num_folds: int = 5, test_pct=0.3,
                       random_state: int = 666, extract_test=True):
    indexes = np.array(X.index)
    # TODO: test mask is missing here
    train_idx, val_idx, test_idx = [], [], []
    units = X[identifier]
    for unit in units.drop_duplicates().values:
        unit_mask = np.where(units == unit)[0]
        L = len(indexes[unit_mask])

        if extract_test:
            test_size = int(L * test_pct)
            i = L - test_size
            test_idx.append(indexes[unit_mask][i:])

            L = i

        val_size = int(L // num_folds)  # not overlapping
        train_size = L - val_size

        # fold begin
        i = int(val_size * ifold)

        train = indexes[unit_mask][i:i + train_size]
        val = indexes[unit_mask][i + train_size:i + train_size + val_size]

        train_idx.append(train)
        val_idx.append(val)

    train_mask = np.isin(np.array(indexes), np.concatenate(train_idx))
    val_mask = np.isin(np.array(indexes), np.concatenate(val_idx))

    if extract_test:
        test_mask = np.isin(np.array(indexes), np.concatenate(test_idx))
    else:
        test_mask = None

    return train_mask, val_mask, test_mask


def __normalize_output_by_unit(X, task):
    if len(task['identifier']) == 0:
        _max = X[task['target']].max()
    else:
        _max = X.groupby(task['identifier'])[task['target']].transform('max')

    X.loc[:, task['target']] = X[task['target']] / _max

    return X



def __split_for_classification(X, fold, num_folds, task, random_state=666, test_pct=0.3):
    # TODO: extract_test handling
    if (len(task['identifier']) > 0) or (
            ('split_by' in task) and (len(task['split_by']) > 0)):  # some dataset have no experiment / units identifier

        if 'split_by' in task:
            train_mask, val_mask, test_mask = __kfold_split_by(X, task['split_by'], fold, num_folds,
                                                               random_state=random_state,
                                                               test_pct=test_pct)

        else:
            split_cols = task['identifier']
            _units = X[split_cols].values

            possible_target_balancing = (
                    X.groupby(split_cols)[task['target']].unique().map(lambda x: len(x)) == 1).all()
            if ('classification' in task['type']):
                logging.info(f"It is possible stratified split? {possible_target_balancing}")

            if ('classification' in task['type']) and possible_target_balancing:  # balance the folds
                assert (X.groupby(split_cols)[task['target']].unique().map(lambda x: len(x)) == 1).all()
                aux = X.groupby(split_cols)[task['target']].min().reset_index()
                units = aux[split_cols].values
                targets = aux[task['target']].values

            else:
                units = list(X[split_cols].drop_duplicates().values)
                targets = None

                # units = _units

            # TODO: incorrect false body of conditional statement
            split_by_unit = (task['num_units'] - 1
                             if 'min_units_per_class' not in task
                             else task['min_units_per_class'] - 1) > 1

            if split_by_unit:
                train_units, val_units, test_units = __stratified_kfold_split_by_unit(units, fold, num_folds,
                                                                                      targets=targets,
                                                                                      random_state=random_state,
                                                                                      test_pct=test_pct)

                val_mask = np.array([np.all(_units == val_units[i], axis=1)
                                     for i in range(len(val_units))]).any(axis=0)
                test_mask = np.array([np.all(_units == test_units[i], axis=1)
                                      for i in range(len(test_units))]).any(axis=0)
                train_mask = (~val_mask) & (~test_mask)
            else:
                train_mask, val_mask, test_mask = __ts_split_by_unit(X, task['identifier'], fold, num_folds,
                                                                     random_state=random_state,
                                                                     test_pct=test_pct)


    else:  # make time series split
        task['identifier'] = ['id']
        X['id'] = 0

        train_mask, val_mask, test_mask = __ts_split_by_unit(X, task['identifier'], fold, num_folds,
                                                             random_state=random_state,
                                                             test_pct=test_pct)

    return test_mask, train_mask, val_mask


def __split_for_regression(X, fold, num_folds, task, extract_test: bool = True, random_state=666, test_pct=0.3):
    if (len(task['identifier']) > 0) or (
            ('split_by' in task) and (len(task['split_by']) > 0)):  # some dataset have no experiment / units identifier

        if 'split_by' in task:
            train_mask, val_mask, test_mask = __kfold_split_by(X, task['split_by'], fold, num_folds, extract_test,
                                                               random_state=random_state,
                                                               test_pct=test_pct)

        else:
            split_cols = task['identifier'] if 'split_by' not in task else task['split_by']
            _units = X[split_cols].values

            if extract_test:
                split_by_unit = (task['num_units'] - 1 if 'min_units_per_class' not in task else task[
                                                                                                     'min_units_per_class'] - 1) > 1
            else:
                split_by_unit = (task['num_units'] if 'min_units_per_class' not in task else task[
                    'min_units_per_class']) > 1

            if split_by_unit:
                train_mask, val_mask, test_mask = __kfold_split_by(X, split_cols, fold, num_folds, extract_test,
                                                                   random_state=random_state,
                                                                   test_pct=test_pct)
            else:
                train_mask, val_mask, test_mask = __ts_split_by_unit(X, task['identifier'], fold, num_folds,
                                                                     extract_test=extract_test,
                                                                     test_pct=test_pct)

    else:  # make time series split
        task['identifier'] = ['id']
        X['id'] = 0

        train_mask, val_mask, test_mask = __ts_split_by_unit(X, task['identifier'], fold, num_folds,
                                                             extract_test=extract_test,
                                                             test_pct=test_pct)

    return test_mask, train_mask, val_mask


def __get_reader(module: str, reader: str):
    dataset_module = importlib.import_module(
        "phmd.readers." + module)  # __import__("phmd.readers." + module)

    return getattr(dataset_module, reader)


def _get_task_key(task):
    return 'target_id' if 'target_id' in task else 'target'


def _get_task(meta, task):
    tasks = list([t for k, t in meta['tasks'].items() if t[_get_task_key(t)] == task])
    if len(tasks) == 0:
        raise Exception(f"Task {task} not found in dataset")
    # task = [t for k, t in meta['tasks'].items() if t[get_task_key(t)] == task][0]
    return tasks[0]


def __validate_values(dataset_name: str, task: str = None, filters: dict = None):
    meta = read_meta(dataset_name)

    # check for valid task
    if task is not None:

        task_ids = [t[_get_task_key(t)] for k, t in meta['tasks'].items()]
        if task not in task_ids:
            task_ids = [str(tid) for tid in task_ids]
            raise ValueError('Task \"%s\" not found. Valid task for %s are: %s' %
                             (task, dataset_name, ', '.join(task_ids)))

    # check valid filters
    if filters is not None:
        if 'filters' not in meta:
            raise ValueError('The dataset %s not define filters. Set to None the filters parameter.' % dataset_name)

        valid_filters = list(meta['filters'].keys())
        for k, values in filters.items():

            if k not in valid_filters:
                raise ValueError('Filter \"%s\" not found. Valid filters are: %s' %
                                 (k, ', '.join(valid_filters)))

            valid_values = meta['filters'][k]
            values = ensure_iterable(values)
            for value in values:
                if value not in valid_values:
                    valid_values = [str(v) for v in valid_values]
                    raise ValueError('Value \"%s\" not found. Valid values for filter %s are: %s' %
                                     (str(value), k, ', '.join(valid_values)))


def __filter_columns(X, meta, task):
    if isinstance(X, list) or isinstance(X, tuple):
        return [__filter_columns(x, meta, task) for x in X]
    else:
        target = task['target']
        if not isinstance(target, list):
            target = [target]

        cols = task['identifier'] + task['features'] + target
        if 'split_by' in task:
            cols += task['split_by']

        cols = list(set(cols))
        drop_cols = [c for c in X.columns if c not in cols]
        X.drop(drop_cols, axis=1, inplace=True)
        # X = X[cols]

        return X


def __load_dataset(dataset_name: str, ftype: str, cache_dir: str = None, task: str = None, filters: dict = None,
                   params: dict = {}, filter_cols=True):
    __validate_values(dataset_name, task, filters)

    meta = read_meta(dataset_name)
    files = meta['files']
    X = None
    for file in files:

        if file['type'] == ftype:
            if task is not None:
                _task = _get_task(meta, task)
            freader = __get_reader(dataset_name.lower(), file['reader'])
            data_dir = get_storage_dir(cache_dir)
            if os.path.exists(os.path.join(data_dir, file["unzipped_dir"])):
                path = os.path.join(data_dir, file['unzipped_dir'])
            else:
                path = os.path.join(data_dir, file['name'])

            if 'params' in meta:
                _params = meta['params']
                _params.update(params or {})
                params = _params

            kwars = {}
            if 'params' in inspect.getfullargspec(freader).args:
                kwars['params'] = params

            _X = freader(path, task=_task, filters=filters, **kwars)

            if filter_cols:
                if isinstance(_X, pd.DataFrame):
                    _X = __filter_columns(_X, meta, _task)
                elif isinstance(_X, tuple): # returned (train,test)
                    pass
                else:
                    _X = [__filter_columns(x, meta, _task) for x in _X]

                    if len(_X) == 1:
                        _X = _X[0]

            if X is None:
                X = _X
            elif isinstance(_X, tuple):
                X = _X
            else:
                X = X.append(_X, ignore_index=True)

    return X

def read_meta(dataset_name: str):
    """
    Reads the metadata for a specified dataset from a JSON file.

    Parameters:
    -----------
    dataset_name : str
        The name of the dataset for which metadata is to be retrieved.

    Returns:
    --------
    dict
        A dictionary containing the metadata of the specified dataset.

    Raises:
    -------
    ValueError
        If the metadata file for the specified dataset does not exist.

    Example:
    --------
    To read the metadata for a dataset named "CWRU":
    ```python
    metadata = read_meta("CWRU")
    ```

    Notes:
    ------
    - The metadata is expected to be in a JSON format located in the `metadata` directory,
      named as `<dataset_name>.json`.
    - The function raises an error if the specified dataset name does not correspond to any existing
      metadata file.
    """

    meta_file = os.path.dirname(os.path.realpath(__file__)) + "/metadata/%s.json" % dataset_name

    if not os.path.exists(meta_file):
        raise ValueError("Dataset \"%s\" unkown." % dataset_name)

    return json.load(open(meta_file, "r"))


def load(dataset_name: str, task: str, cache_dir: str = None,
         filters: dict = None, params: dict = None, force_download: bool = False, unzip: bool = False):
    """
    Loads specified subsets of a dataset, with options for task-specific filtering, caching, and optional download/unzip operations.

    Parameters:
    -----------
    dataset_name : str
        Name of the dataset to load.
    task : str, optional
        Specific task name within the dataset for targeted loading and filtering (default is `None`).
    cache_dir : str, optional
        Directory to cache the dataset locally (default is `None`).
    filters : dict, optional
        Dictionary of filters to apply during dataset loading, where keys are column names and values are conditions (default is `None`).
    params : dict, optional
        Additional parameters for loading or filtering (default is `None`).
    force_download : bool, optional
        If `True`, forces redownloading of the dataset even if cached (default is `False`).
    unzip : bool, optional
        If `True`, unzips the dataset if compressed (default is `False`).
    random_state : int, optional
        Seed for any randomness in data loading (default is 666).

    Returns:
    --------
    tuple or pd.DataFrame
        A tuple containing DataFrames for each specified subset (e.g., ('train', 'val', 'test')),
        or a single DataFrame if only one subset is requested and the task does not require splitting.

    Raises:
    -------
    AttributeError
        If any specified dataset subset is invalid (i.e., not found in the available subsets of the dataset).

    Example:
    --------
    To load the training and validation sets of a specific dataset with filters:
    ```python
    train_val_data = load('my_dataset', datasets=['train', 'val'], filters={'column_name': 'condition'})
    ```

    Notes:
    ------
    - The function first verifies the existence of specified subsets in the dataset metadata.
    - If `force_download` is `True`, the dataset will be downloaded even if cached locally.
    - If the dataset's task requires splitting, only the first subset is loaded; otherwise, all requested subsets are loaded.
    - Timing information for the loading operation is logged.
    """
    start_time = time.time()

    meta = read_meta(dataset_name)
    datasets = [f['type'] for f in meta['files']]

    dtypes = set(f['type'] for f in meta['files'])
    invalid_dtypes = set(datasets) - dtypes

    if len(invalid_dtypes) > 0:
        raise AttributeError(
            f"The subsets {','.join(invalid_dtypes)} were not found. Valid subset are {','.join(dtypes)}")

    # download if it was not
    citation_info = download(dataset_name, cache_dir=cache_dir, force=force_download, unzip=unzip)

    if not citation_info:
        show_citation_info(meta)

    _task = _get_task(meta, task)

    if ('split_data' in _task) and (_task['split_data']):
        result = __load_dataset(dataset_name, datasets[0], cache_dir=cache_dir, task=task, filters=filters, params=params)
    else:
        result = tuple(
            __load_dataset(dataset_name, dataset, cache_dir=cache_dir, task=task, filters=filters, params=params) for
            dataset
            in datasets)

    end_time = time.time()

    logging.info(f"Read in {end_time - start_time} seconds")

    return result

def load_cv_sets(dataset_name: str, task_name, fold: int, num_folds: int = 5, preprocess=None,
                 return_test=False, normalize_output=False, filters=None, random_state=666, test_pct=0.3):
    """
    Loads and preprocesses training, validation, and optionally test datasets for a specified task,
    with the option to normalize outputs and split the data for cross-validation.

    Parameters:
    -----------
    dataset_name : str
        Name of the dataset from which to load data.
    task_name : str
        Name of the task within the dataset, used to select features and target variable.
    fold : int
        The current fold index for cross-validation.
    num_folds : int, optional
        Total number of folds for cross-validation (default is 5).
    preprocess : transformer or None, optional
        Transformer for data preprocessing; if provided, it is fit on training data and applied
        to all sets (default is `None`).
    return_test : bool, optional
        If `True`, includes a test set in the returned dictionary (default is `False`).
    normalize_output : bool, optional
        If `True`, normalizes the output data based on the task-specific normalization rules (default is `False`).
    filters : dict or None, optional
        Dictionary of filters to apply when loading the dataset (default is `None`).
    random_state : int, optional
        Seed for data splitting (default is 666).
    test_pct : float, optional
        Percentage of data to reserve for testing if `return_test` is `True` (default is 0.3).

    Returns:
    --------
    dict
        A dictionary containing DataFrames for 'train', 'val', and optionally 'test' sets,
        each corresponding to a split of the dataset based on `fold` and `num_folds` settings.
        Each DataFrame includes the features and target variable for the specified task.

    Example:
    --------
    To load and split data for a classification task:
    ```python
    data_splits = load_train_sets('my_dataset', 'classification_task', fold=0)
    ```

    Notes:
    ------
    - The function reads dataset metadata to identify subsets (`sets`) and task details.
    - For classification tasks, data is split according to class proportions; for regression tasks,
      it is split according to predefined rules or `test_pct`.
    - If `preprocess` is provided, it is fit on the training set and applied to all sets.
    - If `normalize_output` is `True`, the output (target variable) is normalized according to task-specific settings.
    - The function asserts no missing values in loaded datasets to ensure data integrity.

    """
    # Read metadata for the specified dataset
    meta = read_meta(dataset_name)
    task = _get_task(meta, task_name)  # Retrieve task-specific details
    sets = [f['type'] for f in meta['files']]

    # Load the dataset(s) based on the provided parameters
    if len(sets) > 1:
        X, X_test = load(dataset_name, task=task_name, unzip='True', filters=filters)
        if normalize_output:
            X = __normalize_output_by_unit(X, task)
            X_test = __normalize_output_by_unit(X_test, task)
    else:
        X = load(dataset_name,  task=task_name, unzip='True', filters=filters)
        if isinstance(X, tuple) and len(X) == 2:
            X, X_test = X

            if normalize_output:
                X = __normalize_output_by_unit(X, task)
                X_test = __normalize_output_by_unit(X_test, task)
        else:
            X = X[0]
            X_test = None
            if normalize_output:
                X = __normalize_output_by_unit(X, task)

    # Ensure there are no missing values in the datasets
    assert not X.isnull().any().any()
    if X_test is not None:
        assert not X_test.isnull().any().any()

    # Convert column names to strings
    X.rename(str, axis="columns", inplace=True)
    if X_test is not None:
        X_test.rename(str, axis="columns", inplace=True)

    # Split the data into training, validation, and test sets based on task type
    if 'classification' in task["type"]:
        test_mask, train_mask, val_mask = __split_for_classification(X, fold, num_folds, task,
                                                                     random_state=random_state,
                                                                     test_pct=test_pct)
    else:
        test_mask, train_mask, val_mask = __split_for_regression(X, fold, num_folds, task,
                                                                 extract_test=X_test is None,
                                                                 random_state=random_state,
                                                                 test_pct=test_pct)

    # Apply preprocessing to the data if a preprocessing function is provided
    features = task['features']
    if preprocess is not None:
        preprocess.fit(X[train_mask][features])
        X.loc[:, features] = preprocess.transform(X.loc[:, features])

        if X_test is not None:
            X_test.loc[:, features] = preprocess.transform(X_test.loc[:, features])

    # Prepare the sets to return
    _sets = {
        'train': X[train_mask],  # Training set
        'val': X[val_mask]  # Validation set
    }
    if return_test:
        if X_test is None:
            X_test = X[test_mask]

        _sets['test'] = X_test  # Test set, if required

    return _sets



def __print_info(key: str, meta: dict, title: str):
    if key in meta:
        print("\n%s" % title)
        print("=" * (len(title)))
        print_dict(meta[key])


def describe(dataset_name: str):
    """
        Displays a detailed description of a dataset, including its main metadata attributes and additional
        information sections.

        Parameters:
        -----------
        dataset_name : str
            The name of the dataset to describe. This name is used to locate and load the dataset's metadata,
            which is expected to be accessible through the `read_meta` function.

        Behavior:
        ---------
        The function retrieves metadata for the specified dataset and outputs it in a structured format.
        Key attributes and sections are printed, including:

        - General information: The dataset's description, designation, publisher, domain, and application.
        - System Info: Information specific to the dataset's system, if available.
        - Features: A list of features or variables in the dataset.
        - Operating Conditions: Conditions under which data was collected.
        - Tasks: Description of tasks supported by the dataset.
        - Resources: Additional resources or files associated with the dataset.
        - References: Relevant references for the dataset (papers, versions, BibTeX), excluding fields such as
          'papers', 'ver', and 'bibitex' if present.

        Example:
        --------
        To describe a dataset called "example_dataset":
        ```python
        >>> describe("CWRU")

        Description
        ===========
        In this renowned dataset, experiments were conducted utilizing a 2 HP Reliance Electric motor, where acceleration data was measured at locations both near to and remote from the motor bearings. Motor bearings were intentionally seeded with faults using electro-discharge machining (EDM). These faults ranged in diameter from 0.007 inches to 0.040 inches and were separately introduced at the inner raceway, rolling element (i.e., ball), and outer raceway of the bearings. After seeding, the faulted bearings were reinstalled into the test motor, and vibration data was recorded for motor loads ranging from 0 to 3 horsepower, corresponding to motor speeds between 1797 and 1720 RPM. The vibration data was sampled at a frequency of 12 kHz, with each vibration signal lasting for a duration of 10 seconds.

        Designation: Bearing Fault Diagnostic
        Publisher: Case Western Reserve University
        Domain: Mechanical component
        Application: Bearing

        System info
        ===========
        1. type    : Rotatory machine :  bearing
        2. sensors : Voltmeter, ammeter and thermocouple sensor suite
        3. bearing : 6205-2RSL JEM SKF deep-groove ball bearing (and NTN equivalent)
        ...
        ...
        ...

        ```

        Notes:
        ------
        - This function assumes the existence of a `read_meta` function to retrieve the dataset's metadata.
        - The function uses a helper, `__print_info`, to format and display sections if they are available in the metadata.

        """
    meta = read_meta(dataset_name)
    print("%s\n" % dataset_name)
    print("Description")
    print("=" * 11)
    print(meta['description'])
    print()
    print("Designation:", meta['designation'])
    print("Publisher:", meta['publisher'])
    print("Domain:", meta['domain'])
    print("Application:", meta['application'])
    print("License:", meta['license'])

    __print_info('system', meta, 'System info')
    __print_info('features', meta, 'Features')
    __print_info('operating_conditions', meta, 'Operating Conditions')
    __print_info('tasks', meta, 'Tasks')
    __print_info('filters', meta, 'Filters')
    __print_info('resources', meta, 'Resources')
    if 'papers' in meta['references']:
        del meta['references']['papers']
    if 'ver' in meta['references']:
        del meta['references']['ver']
    if 'bibitex' in meta['references']:
        del meta['references']['bibitex']

    __print_info('references', meta, 'References')


def search(**filters):
    """
    Searches datasets in the local 'datasets' directory based on given filter criteria, displaying or returning
    results with metadata for each matched dataset. The function loads JSON metadata files, organizes relevant
    dataset attributes, applies the specified filters, and outputs the filtered list.

    Parameters:
    -----------
    **filters : keyword arguments
        Arbitrary filters that can be applied to refine the search results based on dataset attributes.

        - `return_names` (bool): If `True`, the function will return a set of dataset names that match the filters.
          Default is `False`, in which case a table of results will be printed.

        - Other filters (str): Keys in the metadata attributes (such as "domain", "nature", etc.) that can be
          matched against specific values, allowing users to filter datasets accordingly. For example:
          `search(domain="biology", nature="temporal")`.

    Returns:
    --------
    set or None
        If `return_names` is `True`, returns a set of matching dataset names. If `return_names` is `False`,
        prints a table of the filtered datasets, with columns:

        - `name`: The dataset name.
        - `domain`: The dataset's domain.
        - `nature`: The task's nature (e.g., "temporal", "static").
        - `application`: The application domain of the dataset.
        - `task [target var]`: The task name with its target variable in brackets.
        - `data nature`: The data nature for the task.
        - `features`: A comma-separated list of feature types.

    Example:
    --------
    To search for datasets in the "Drive technology" domain and print by screen:
    ```python
    >>> search(domain="drive")

            name     domain            nature       application    task [target var]                                               data nature    features
    --  -------  ----------------  -----------  -------------  --------------------------------------------------------------  -------------  ---------------------------------------------------------------
     8  CBM14    Drive technology  time-series  Naval          GS Compressor decay state coefficient [gcdsc]                   time-series    speed,torque,flow,temperature,pressure,position
     9  CBM14    Drive technology  time-series  Naval          GT Turbine decay state coefficient [gtdsc]                      time-series    speed,torque,flow,temperature,pressure,position
    10  CBMv3    Drive technology  time-series  Naval          Propeller Thrust decay state coefficient (Kkt) [ptdsc_port]     time-series    speed,torque,temperature,pressure,flow
    11  CBMv3    Drive technology  time-series  Naval          2. Propeller Torque decay state coefficient (Kkq) [ptdsc_stbd]  time-series    speed,torque,temperature,pressure,flow
    12  CBMv3    Drive technology  time-series  Naval          3. Hull decay state coefficient (Khull) [hdsc]                  time-series    speed,torque,temperature,pressure,flow
    13  CBMv3    Drive technology  time-series  Naval          4. GT Compressor decay state coefficient (KMcompr) [gcdsc]      time-series    speed,torque,temperature,pressure,flow
    14  CBMv3    Drive technology  time-series  Naval          5. GT Turbine decay state coefficient (KMturb) [gtdsc]          time-series    speed,torque,temperature,pressure,flow
    17  CMAPSS   Drive technology  time-series  Aircraft       Prognosis [rul]                                                 time-series    speed,temperature,angle,pressure,altitude,flow
    18  CMAPSS   Drive technology  time-series  Aircraft       Diagnosis [Fault_type]                                          time-series    speed,categorical,temperature,angle,pressure,altitude,flow
    50  NCMAPSS  Drive technology  time-series  Aircraft       Prognosis [rul]                                                 time-series    speed,bool,categorical,temperature,angle,pressure,altitude,flow
    65  PHMAP23  Drive technology  time-series  Spacecraft     Prognosis [fault]                                               time-series    pressure
    67  PHME24   Drive technology  time-series  Servomotor     RUL [rul]                                                       time-series    current,voltage,temperature,boolean,time,velocity,position
    ```

    To search for datasets in the "Drive technology" domain and retrieve names only:
    ```python
    >>> search(domain="drive", return_names=True)

    {'CBMv3', 'NCMAPSS', 'CBM14', 'PHME24', 'CMAPSS', 'PHMAP23'}
    ```

    Notes:
    ------
    - This function requires that JSON metadata files for datasets are stored in a "metadata" directory
      in the same directory as the script.
    - Filters are non case-insensitive and partially matched (i.e., a filter of "bio" would match "biology").

    """
    return_names = False
    if 'return_names' in filters:
        return_names = filters['return_names']
        del filters['return_names']

    _dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "metadata")

    sets = [f.replace('.json', '') for f in os.listdir(_dir)
            if 'template' not in f and 'excluded' not in f and '.json' in f]

    data = {
        'name': [],
        'domain': [],
        'nature': [],
        'application': [],
        'task [target var]': [],
        'data nature': [],
        'features': [],
        'publisher': []
    }
    for dataset_name in sorted(sets):
        meta = read_meta(dataset_name)

        for task_name, task in meta['tasks'].items():
            data['publisher'].append(meta['publisher'])
            data['name'].append(dataset_name)
            data['domain'].append(meta['domain'])
            data['nature'].append(task['nature'])
            data['application'].append(meta['application'])
            data['task [target var]'].append(f"{task_name} [{task['target']}]")
            data['data nature'].append(task['nature'])
            if len(meta['features']) == 0:
                features = ["Unkown"]
            else:
                features = {k.lower(): v for k, v in meta['features'].items()}
                features = [features[f.lower()] for f in task['features'] if f.lower() in features]
                features = set([f['type'].lower() for f in features if f['type'] != ""])
            data['features'].append(",".join(features))

    df = pd.DataFrame(data)
    del data['publisher']
    for col, filter in filters.items():
        if col == 'task':
            col = 'task [target var]'

        df = df[df[col].str.lower().str.contains(filter.lower())]

    if return_names:
        return set(list(df.name.values))
    else:
        print(tabulate(df, headers='keys', tablefmt='plano', showindex=False))

        print(f"Found {df.shape[0]} datasets")


class Dataset:
    def __init__(self, dataset_name: str, cache_dir: Optional[str] = None):
        self.dataset_name = dataset_name
        self.cache_dir = cache_dir or get_storage_dir()
        self.meta = self._load_meta()
        self.tasks = [Task(self, task[_get_task_key(task)]) for task_name, task in self.meta['tasks'].items()]

    def _load_meta(self):
        return read_meta(self.dataset_name)

    def download(self, force: bool = False, unzip: bool = False):
        download(self.dataset_name, cache_dir=self.cache_dir, force=force, unzip=unzip)

    def describe(self):
        describe(self.dataset_name)

    @staticmethod
    def search(**filter):
        return search(**filter)

    def get_task(self, task_name: str) -> "Task":
        for task in self.tasks:
            if (task.target == task_name) or (task.name == task_name):
                return task
        valid_tasks = ','.join([task[_get_task_key(task)] for _, task in self.meta['tasks'].items()])
        raise ValueError(f"Task '{task_name}' not found in dataset '{self.dataset_name}'. Valid tasks are: {valid_tasks}.")

    def __getitem__(self, index: Union[int, str]) -> "Task":
        if isinstance(index, int):
            return self.tasks[index]
        return self.get_task(index)

    def __iter__(self) -> Iterator["Task"]:
        return iter(self.tasks)

    def __len__(self) -> int:
        return len(self.tasks)


class Task:
    def __init__(self, dataset: Dataset, task_name: str):
        self.dataset = dataset
        self.name = task_name
        self.meta = _get_task(self.dataset.meta, task_name)
        self.features = self.meta["features"]
        self.target = _get_task_key(self.meta)

        if len(self.meta['identifier']) == 0:
            self.max_folds = 0
        else:
            self.max_folds = (self.meta['num_units'] - 1
                         if 'min_units_per_class' not in self.meta
                         else self.meta['min_units_per_class'] - 1)

        if self.max_folds <= 1:  # split time series
            self.max_folds = 3

        self.folds = min(5, self.max_folds)
        self.preprocess = None
        self.normalize_output = False
        self.test_pct = 0.3
        self.return_test = True
        self.random_state = 666

    def load_fold(self, fold: int):
        if fold < 0 or fold >= self.folds:
            raise ValueError(f"Fold must be between 0 and {self.folds - 1}.")

        sets = load_cv_sets(
            dataset_name=self.dataset.dataset_name,
            task_name=self.name,
            fold=fold,
            num_folds=self.folds,
            preprocess=self.preprocess,
            normalize_output=self.normalize_output,
            return_test=self.return_test,
            test_pct=self.test_pct
        )

        set_keys = ",".join(list(sets.keys()))
        logging.info(f"Read {len(sets)} sets: {set_keys}")
        columns = ",".join(list(sets['train'].columns))
        logging.info(f"Columns: {columns}")
        for key, df in sets.items():
            logging.info(f"{key.capitalize()} shape: {df.shape}")

        return sets

    def __getitem__(self, fold: int):
        return self.load_fold(fold)

    def __iter__(self) -> Iterator[pd.DataFrame]:
        for fold in range(self.folds):
            yield self.load_fold(fold)

    def __len__(self) -> int:
        return self.folds

