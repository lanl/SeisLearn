from pathlib import Path
import os
from functools import lru_cache, wraps
from datetime import datetime,timedelta

def get_repo_dir():
    file_dir = Path(os.path.dirname(os.path.realpath(__file__)))
    repo_dir = file_dir.parents[1]
    return repo_dir


def get_data_dir():
    file_dir = Path(os.path.dirname(os.path.realpath(__file__)))
    data = f"{file_dir.parents[2]}/Data"
    return data


def timed_lru_cache(seconds:int, maxsize:int=None, verbose:bool=False):
    def wrapper_cache(func):
        func = lru_cache(maxsize=maxsize)(func)
        func.lifetime = timedelta(seconds=seconds)
        func.expiration = datetime.now() + func.lifetime
        func.verbose = verbose

        @wraps(func)
        def wrapped_func(*args, **kwargs):
            if func.verbose:
                print(f"Current Time: {datetime.now()}, Cache expiration: {func.expiration}")
            if datetime.now() >= func.expiration:
                if func.verbose:
                    print("Cache lifetime expired, retrieving data")
                func.cache_clear()
                func.expiration = datetime.now() + func.lifetime
            
            return func(*args, **kwargs)
        
        return wrapped_func
    
    return wrapper_cache


def iou_datasets(rand_list, sparse_list):
    """
    Calculate IoU for the random and sparse sampling datasets
    """
    set_rand = set(rand_list)
    set_sprs = set(sparse_list)

    intersection = set_rand.intersection(set_sprs)
    union = set_rand.union(set_sprs)

    iou = len(intersection) / len(union) if union else 0

    return iou

