# -*- coding: utf-8 -*-
import os
import json
import shutil
import tarfile
from tqdm import tqdm
import requests
from phmd.utils import get_storage_dir
from phmd import *
import zipfile
import hashlib


def download_from_common_server(url: str, fname: str):
    resp = requests.get(url, stream=True, timeout=5)
    total = int(resp.headers.get('content-length', 0))
    with open(fname, 'wb') as file, tqdm(
            desc=fname,
            total=total,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)


def download_file(url: str, target: str):
    try:
        download_from_common_server(url, target)
        return True

    except Exception as ex:
        print(ex)
        return False


def _extract_archive(file_path, path='.'):
    """Extracts an archive if it matches zip format.
    Args:
      file_path: path to the archive file
      path: path to extract the archive file
    Returns:
      True if a match was found and an archive extraction was completed,
      False otherwise.
    """

    open_fn = zipfile.ZipFile
    is_match_fn = zipfile.is_zipfile

    if is_match_fn(file_path):
        with open_fn(file_path) as archive:
            try:
                archive.extractall(path)
            except (tarfile.TarError, RuntimeError, KeyboardInterrupt):
                if os.path.exists(path):
                    if os.path.isfile(path):
                        os.remove(path)
                    else:
                        shutil.rmtree(path)
                raise
        return True

    return False

def _get_hash(fpath, chunk_size=65535):

    hasher = hashlib.md5()
    with open(fpath, 'rb') as fpath_file:
        for chunk in iter(lambda: fpath_file.read(chunk_size), b''):
            hasher.update(chunk)

    return hasher.hexdigest()



def _checkhash(fpath, md5_hash):
    return _get_hash(fpath) == md5_hash

def __unzip_handling(unzip, _id, file_path, datadir):
    if unzip:
        print("Extracting files %s..." % _id)
        _extract_archive(file_path, datadir)

def __validate_handling(check_hash, file_path, md5_hash):
    if not os.path.exists(file_path):
        return False
    elif check_hash:
        dhash = _get_hash(file_path)
        valid = dhash == md5_hash
        if not valid:
            print("File %s not valid (invalid md5 hash), it might be corrupted. It will be removed. MD5sum computed: %s" % (file_path, dhash))
            os.remove(file_path)
        return valid
    else:
        return True


def download(dataset: str, cache_dir: str = None, unzip: bool = False, force: bool = False):
    """
    Downloads the dataset specified by the given ID and handles its extraction.

    Parameters:
    -----------
    dataset : str
        The identifier for the dataset to download.

    cache_dir : str, optional
        The directory where the dataset should be cached. If not specified, a default storage directory is used.

    unzip : bool, optional
        If True, the downloaded dataset will be unzipped. Default is False.

    force : bool, optional
        If True, forces the download even if the dataset already exists in the cache. Default is False.

    Raises:
    -------
    ValueError
        If the dataset identifier does not correspond to any known dataset.

    Example:
    --------
    To download a dataset with the ID "dataset123":
    ```python
    download("dataset123", cache_dir="/path/to/cache", unzip=True)
    ```

    Notes:
    ------
    - The function reads the dataset metadata from a JSON file located in the `metadata` directory.
    - It uses the metadata to obtain download URLs and MD5 checksums for validation.
    - If the dataset already exists and `force` is not set to True, it will skip downloading unless the
      existing file is corrupted (based on MD5 validation).
    - Citing the original publisher of the dataset is encouraged, with relevant citation information printed
      during the download process.
    """
    meta_file = os.path.dirname(os.path.realpath(__file__)) + "/metadata/%s.json" % dataset
    if not os.path.exists(meta_file):
        raise ValueError("Dataset \"%s\" unkown." % dataset)

    meta = json.load(open(meta_file, "r"))

    datadir = get_storage_dir(cache_dir)
    os.makedirs(datadir, exist_ok=True)

    citation_info = False

    files = meta["files"]
    for file in files:
        file_path = os.path.join(datadir, file["name"])
        if not os.path.exists(file_path) or force or not __validate_handling(True, file_path, md5sum(file)):   # check if exists the zip
            if not citation_info:
                show_citation_info(meta)
                citation_info = True
            downloaded = download_file(get_download_url(file), file_path) and \
                         __validate_handling(True, file_path, md5sum(file))
            if downloaded:
                __unzip_handling(unzip, dataset, file_path, datadir)

        else:
            exists = os.path.exists(os.path.join(datadir, file['unzipped_dir']))
            if unzip:
                if exists:
                    print("Dataset %s already downloaded and extracted" % dataset)
                else:
                    print("Unzipping...")
                    __unzip_handling(unzip, dataset, file_path, datadir)
            elif not exists:
                print("Dataset %s already downloaded but not unzipped" % dataset)

    return citation_info


