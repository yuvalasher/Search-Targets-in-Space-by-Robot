from typing import Tuple, Any
from pathlib import Path
import pickle
import h5py
import numpy as np

Location = Tuple[int, int]
CONFIG_PATH: Path = Path('config.cfg')
NOISE: float = 1e-8

# Early Stopping Params
MIN_IMPROVEMENT: float = 1e-3
PATIENT_NUM_EPOCHS: int = 100
PRINT_EVERY: int = 1


def save_pickle_object(obj_name: str, obj: Any) -> None:
    with open(f'Pickles/{obj_name}.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_pickle_object(obj_name: str) -> Any:
    with open(f'Pickles/{obj_name}.pkl', 'rb') as f:
        return pickle.load(f)


def save_hdf5_to_file(obj_name: str, obj: object) -> None:
    with h5py.File(f'hdf5/{obj_name}.h5', 'w') as hf:
        hf.create_dataset(obj_name, data=obj)


def load_hdf5_file(obj_name: str) -> Any:
    with h5py.File(f'hdf5/{obj_name}.h5', 'r') as hf:
        return hf[obj_name][:]


def check_earlystopping(loss: np.array, epoch: int, min_improvement: float = MIN_IMPROVEMENT,
                        patient_num_epochs: int = PATIENT_NUM_EPOCHS) -> bool:
    """
    Checking convergence in patient_num_epochs before to check if there is still loss improvement in the loss by at
    minimum min_improvement.
    This should be applied on a validation loss, hence, it can cause overfitting on the test set.
    """
    if epoch > patient_num_epochs:
        return np.sum(np.where((loss[epoch - 1 - patient_num_epochs:epoch - 1] -
                                loss[epoch - patient_num_epochs:epoch]) >= min_improvement, 1, 0)) == 0
