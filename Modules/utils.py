from typing import Tuple
from pathlib import Path
import pickle

Location = Tuple[int, int]
CONFIG_PATH = Path('config.cfg')


def save_pickle_object(obj_name: str, obj: object):
    with open('Pickles/{}.pkl'.format(obj_name), 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_pickle_object(obj_name: str):
    with open('Pickles/{}.pkl'.format(obj_name), 'rb') as f:
        return pickle.load(f)
