from typing import Tuple, Any
import pickle
import h5py
import os
import numpy as np
from matplotlib import pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, recall_score, precision_score
from consts import FILES_PATH, MIN_IMPROVEMENT, PATIENT_NUM_EPOCHS, TRAIN_RATIO, VALIDATION_RATIO, BATCH_SIZE


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


def convert_probs_to_preds(probs: np.array, threshold: float = 0.5) -> np.array:
    """
    Convert probabilities into labels by a given threshold. Probabilities above the threshold will be 1, otherwise 0.
    """
    return np.where(probs >= threshold, 1, 0)


def get_num_of_areas_and_targets_from_arary(array: np.array, verbose: bool = True) -> Tuple[int, int]:
    """
    Calculate number of different Areas the array is working on and how many targets in total
    """
    num_of_areas, num_of_targets = array.shape[0], array.sum()
    if verbose:
        print(f'Num of Areas: {num_of_areas}; Num of real targets: {int(num_of_targets)}')
    return num_of_areas, num_of_targets


def calculate_model_metrics(y_true: np.array, y_pred: np.array, verbose: bool = True, mode: str = 'Test') -> Tuple[
    float, float, float]:
    """
    Calculating Accuracy, recall, precision
    """
    if type(y_true) == torch.tensor:
        y_true = y_true.detach().numpy()
    y_pred = convert_probs_to_preds(probs=y_pred)
    y_true, y_pred = y_true.reshape(-1), y_pred.reshape(-1)
    assert y_true.shape == y_pred.shape
    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    recall = recall_score(y_true=y_true, y_pred=y_pred)  # , average='samples')
    precision = precision_score(y_true=y_true, y_pred=y_pred)  # , average=='samples')
    if verbose:
        print(f'*** {mode} ***')
        print(f'Num of found targets: {(y_pred[np.where(y_true == 1)[0]] == 1).sum()} / {int(y_true.sum())}')
        print(f'Accuracy: {accuracy * 100:.2f}%; Recall: {recall * 100:.2f}%; Precision: {precision * 100:.2f}%')

    return accuracy, recall, precision


def load_data(train_ratio: float = TRAIN_RATIO, validation_ratio: float = VALIDATION_RATIO):
    """
    Loading data from disk
    :return: arrays of X, y for both train, validation and test sets.
    """
    X, y = load_X_y_from_disk()
    x_train, x_val, x_test, y_train, y_val, y_test = split_to_train_validation_test(X=X, y=y,
                                                                                    train_ratio=train_ratio,
                                                                                    validation_ratio=validation_ratio)
    return x_train, x_val, x_test, y_train, y_val, y_test


def load_X_y_from_disk(max_size: int = None) -> Tuple[np.array, np.array]:
    X, y = load_hdf5_file('X_10000_100'), load_hdf5_file('y_10000_100')

    X = X[:, :, :, 0]
    if max_size:
        X = X[:max_size, :, :]
        y = y[:max_size, :]
    return X, y


def split_to_train_validation_test(X: np.array, y: np.array, train_ratio: float = TRAIN_RATIO,
                                   validation_ratio: float = VALIDATION_RATIO) -> \
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
                                y_val: np.array, y_test: np.array, batch_size: int = BATCH_SIZE) -> Tuple[
    DataLoader, DataLoader, DataLoader]:
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


def print_data_statistics(x_train: np.array, x_val: np.array, x_test: np.array, y_train: np.array, y_val: np.array,
                          y_test: np.array) -> None:
    print(x_train.shape, x_val.shape, x_test.shape, y_train.shape, y_val.shape, y_test.shape)
    print(f'Num Train of targets: {int(y_train.sum())}')
    print(f'Num Val of targets: {int(y_val.sum())}')
    print(f'Num Test of targets: {int(y_test.sum())}')
