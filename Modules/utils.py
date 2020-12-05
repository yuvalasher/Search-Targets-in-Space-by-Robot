from typing import Tuple, Any
import pickle
import h5py
import os
import numpy as np
from matplotlib import pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from consts import FILES_PATH, MIN_IMPROVEMENT, PATIENT_NUM_EPOCHS


def save_pickle_object(obj_name: str, obj: Any) -> None:
    with open(f'Pickles/{obj_name}.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_pickle_object(obj_name: str) -> Any:
    if not os.path.exists(f'{os.getcwd()}\Pickles'):
        path = FILES_PATH + '\Pickles'
    else:
        path = 'Pickles'
    with open(path + f'/{obj_name}.pkl', 'rb') as f:
        return pickle.load(f)


def save_hdf5_to_file(obj_name: str, obj: object) -> None:
    with h5py.File(f'hdf5/{obj_name}.h5', 'w') as hf:
        hf.create_dataset(obj_name, data=obj)


def load_hdf5_file(obj_name: str) -> Any:
    if not os.path.exists(f'{os.getcwd()}\hdf5'):
        path = FILES_PATH + "\hdf5"
    else:
        path = 'hdf5'
    with h5py.File(path + f'/{obj_name}.h5', 'r') as hf:
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


def convert_probs_to_preds(probs: np.array, threshold: float = 0.7) -> np.array:
    """
    Convert probabilities into labels by a given threshold. Probabilities above the threshold will be 1, otherwise 0.
    """
    return np.where(probs >= threshold, 1, 0)


def calculate_metrics(y_true: np.array, y_pred: np.array) -> Tuple[float, float, float, float]:
    """
    Calculating Accuracy, recall, precision, f1-score
    """
    y_pred = convert_probs_to_preds(probs=y_pred)
    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    recall = recall_score(y_true=y_true, y_pred=y_pred)
    precision = precision_score(y_true=y_true, y_pred=y_pred)
    f1score = f1_score(y_true=y_true, y_pred=y_pred)
    return accuracy, recall, precision, f1score


def load_X_y_from_disk(num_features: int = 2, max_size: int = None) -> Tuple[np.array, np.array]:
    """
    :param num_features: Decide if taking all the features of just the first (X vector)
    """
    X, y = load_hdf5_file('X'), load_hdf5_file('y')

    if num_features == 1:
        X = X[:, :, :, 0]
        if max_size:
            X = X[:max_size, :, :]
            y = y[:max_size, :]
    else:
        if max_size:
            X = X[:max_size, :, :, :]
            y = y[:max_size, :]
    return X, y


def _split_to_train_validation_test(X: np.array, y: np.array, train_ratio: float = 0.7,
                                    validation_ratio: float = 0.15) -> \
        Tuple[np.array, np.array, np.array, np.array, np.array, np.array]:
    """
    Splitting the data based on the ratio of the different types of data (train, validation, test)
    :param X: The signals received from the cells. Shape: (seq_len, num_cells)
    :param y: The ground truth label of existence of targets in cells. Shape: (N * N,)
    :return: X, y for each dataset type (train, validation, test)
    """
    num_sequences = X.shape[0]
    train_len = int(np.ceil(num_sequences * train_ratio))
    validation_len = int(np.ceil(num_sequences * validation_ratio))
    return X[:train_len], X[train_len:train_len + validation_len], X[train_len + validation_len:], y[:train_len], \
           y[train_len:train_len + validation_len], y[train_len + validation_len:]

class TargetsDataset(Dataset):
    """
    The dataset object used to read the data
    """

    def __init__(self, features: np.array, labels: np.array):
        assert features.shape[0] == labels.shape[0]
        self.features = torch.Tensor(features).to(torch.float32)
        self.labels = torch.Tensor(labels).to(torch.float32)

    def __getitem__(self, idx):
        return self.features[idx, :].float(), self.labels[idx, :].float()

    def __len__(self):
        return self.labels.shape[0]

def get_dataloader_for_datasets(x_train: np.array, x_val: np.array, x_test: np.array, y_train: np.array,
                                y_val: np.array, y_test: np.array, batch_size: int=256) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    The length of the data-loaders are number of samples in X divided to batch size (X.shape[0] / batch_size))
    """
    train_dataloader = DataLoader(TargetsDataset(features=x_train, labels=y_train), batch_size=batch_size)
    val_dataloader = DataLoader(TargetsDataset(features=x_val, labels=y_val), batch_size=batch_size)
    test_dataloader = DataLoader(TargetsDataset(features=x_test, labels=y_test), batch_size=batch_size)
    return train_dataloader, val_dataloader, test_dataloader

def plot_values_by_epochs(train_values: np.array, validation_values: np.array = None, test_values: np.array = None,
                          title: str = 'Loss VS Epoch') -> None:
    """
    Line-plots of 2 sets of values against the epoch value
    """
    fig, ax = plt.subplots()
    ax.plot(list(range(len(train_values))), train_values, label='Train')
    if validation_values is not None:
        ax.plot(list(range(len(validation_values))), validation_values, label='Validation')
    if test_values is not None:
        ax.plot(list(range(len(test_values))), test_values, label='Test')
    ax.legend()
    ax.set_ylim(ymin=0)
    ax.set_title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.show()
