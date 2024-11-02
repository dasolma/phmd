import os 
from hashlib import sha1

def get_hash(**kwargs):
    return sha1(repr(sorted(kwargs.items())).encode()).hexdigest()

def get_storage_dir(cache_dir: str = None):

    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser('~'), '.phmd')
    datadir_base = os.path.expanduser(cache_dir)

    if not os.path.exists(datadir_base):
        try:
            os.makedirs(datadir_base)
        except:
            pass

    if not os.access(datadir_base, os.W_OK):
        datadir_base = os.path.join('/tmp', '.phmd')

    datadir = os.path.join(datadir_base, "datasets")

    return datadir


def print_dict(data: dict, level: int = 0):
    if len(data.keys()) > 0:
        name_len = max(len(k) for k in data.keys())

        for k in sorted(data.keys()):
            v = data[k]
            name_format = "%%%ds" % (name_len - len(k) + 2)
            
            if isinstance(v, dict):
                name = (" " * (level*3)) + k + " :"
                print(name)
                print_dict(v, level+1)
            else:
                if isinstance(v, list):
                    v = ','.join([str(i) for i in v])
                    
                name = (" " * (level*3)) + k + (name_format % ":")
                print(name, v)

