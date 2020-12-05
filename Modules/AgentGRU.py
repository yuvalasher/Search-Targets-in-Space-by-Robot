import torch
from torch import nn
from torch.nn.modules import loss
from torch.utils.data import DataLoader, Dataset
from typing import Tuple, Dict, Any
import numpy as np
from utils import load_hdf5_file, PRINT_EVERY, plot_values_by_epochs
from tqdm import tqdm

train_on_gpu = torch.cuda.is_available()


class AgentGRU(nn.Module):
    def __init__(self, seq_len: int, input_size: int, hidden_dim: int, output_dim: int = 1, num_layers: int = 2,
                 dropout_prob: float = 0.5):
        super(AgentGRU, self).__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_dim, num_layers=num_layers)  # , batch_first=True)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: np.array, hidden: bool = None):
        """
        Forward pass of the net
        """
        x = x.permute(1, 0, 2)
        if not hidden:
            h = self.init_params(x.size(1))
        else:
            h = hidden
        # print(f'x: {x.shape}, h: {h.shape}')
        out, hidden = self.gru(x, h.detach())
        # out = self.dropout(out)
        out = self.fc(out[-1, :, :])  # Fully Connected on the last timestamp
        out = self.relu(out)
        return out, hidden

    def init_params(self, batch_size: int) -> torch.Tensor:
        """
        Init the weights of the hidden states with random
        return matrix of zeros in size (self.num_layers, batch_size, self.hidden_dim)
        """
        if train_on_gpu:
            return torch.zeros(self.num_layers, batch_size, self.hidden_dim).requires_grad_().cuda()
        h = torch.zeros(self.num_layers, batch_size, self.hidden_dim).requires_grad_()
        return h


def train(net: nn.Module, train_dataloader: DataLoader = None, val_dataloader: DataLoader = None,
          test_dataloader: DataLoader = None) -> nn.Module:
    """
    Training loop iterating on the train dataloader and updating the model's weights.
    Inferring the validation dataloader & test dataloader, if given, to babysit the learning
    Activating cuda device if available.
    :return: Trained model
    """
    train_losses: np.array = np.zeros(NUM_EPOCHS)
    test_losses: np.array = np.zeros(NUM_EPOCHS)
    val_losses: np.array = np.zeros(NUM_EPOCHS)
    hidden_list: list = []

    if test_dataloader:
        untrained_test_loss = infer(net, test_dataloader, loss_fn)
        print(f'Test Loss before training: {untrained_test_loss:.3f}')

    for epoch in range(NUM_EPOCHS):
        net.train()
        for x_train, y_train in tqdm(train_dataloader):
            if train_on_gpu:
                net.cuda()
                x_train = x_train.cuda()
                y_train = y_train.cuda()
            y_train_pred, hidden = net(x_train)
            # We don't want to propagate the whole network
            hidden = hidden.data
            hidden_list.append(hidden)

            train_loss = loss_fn(y_train_pred, y_train)
            train_losses[epoch] = train_loss.item() / len(train_dataloader)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            if val_dataloader:
                val_losses[epoch] = infer(net, val_dataloader, loss_fn)

            if test_dataloader:
                test_losses[epoch] = infer(net, test_dataloader, loss_fn)

        if epoch % PRINT_EVERY == 0:
            print(f"Epoch: {epoch + 1}/{NUM_EPOCHS},",
                  f"Train loss: {train_losses[epoch]:.4f},",
                  f"Validation loss: {val_losses[epoch]:.4f}")
            print(f"Test Loss: {test_losses[epoch]:.2f}" if test_dataloader else '')
    plot_values_by_epochs(train_values=train_losses, validation_values=val_losses)
    return net


def infer(net: nn.Module, infer_dataloader: DataLoader, loss_fn: loss) -> float:
    """
    Run the model on x_infer (both validation and test) and calculate the loss of the predictions.
    The model run on evaluation mode and without updating the computational graph (no_grad)
    Running on the dataloader by batches, defined in each dataset's DataLoader
    """
    net.eval()
    running_loss = 0
    for x, y in test_dataloader:
        with torch.no_grad():
            if train_on_gpu:
                x, y = x.cuda(), y.cuda()
            y_pred, hidden = net(x)
            infer_loss = loss_fn(input=y_pred.reshape(-1), target=y.reshape(-1))
        running_loss += infer_loss
    return running_loss / len(infer_dataloader)


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


def load_X_y_from_disk(num_features: int = 2) -> Tuple[np.array, np.array]:
    """
    :param num_features: Decide if taking all the features of just the first (X vector)
    """
    X, y = load_hdf5_file('X'), load_hdf5_file('y')
    if num_features == 1:
        X = X[:, :, :, 0]
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


def get_dataloader_for_datasets(x_train: np.array, x_val: np.array, x_test: np.array, y_train: np.array,
                                y_val: np.array, y_test: np.array) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    The length of the data-loaders are number of samples in X divided to batch size (X.shape[0] / batch_size))
    """
    train_dataloader = DataLoader(TargetsDataset(features=x_train, labels=y_train), batch_size=batch_size)
    val_dataloader = DataLoader(TargetsDataset(features=x_val, labels=y_val), batch_size=batch_size)
    test_dataloader = DataLoader(TargetsDataset(features=x_test, labels=y_test), batch_size=batch_size)
    return train_dataloader, val_dataloader, test_dataloader


if __name__ == '__main__':
    """
    Usage of focal loss - https://discuss.pytorch.org/t/is-this-a-correct-implementation-for-focal-loss-in-pytorch/43327/6
    Try to use bidirectional model
    """
    NUM_EPOCHS: int = 50
    train_ratio: float = 0.7
    validation_ratio: float = 0.15
    test_ratio: float = 0.15
    hidden_dim: int = 32
    num_layers: int = 2
    lr: float = 1e-3
    batch_size: int = 256

    X, y = load_X_y_from_disk(num_features=1)
    x_train, x_val, x_test, y_train, y_val, y_test = _split_to_train_validation_test(X=X, y=y,
                                                                                     train_ratio=train_ratio,
                                                                                     validation_ratio=validation_ratio)

    train_dataloader, val_dataloader, test_dataloader = get_dataloader_for_datasets(x_train=x_train, x_val=x_val,
                                                                                    x_test=x_test, y_train=y_train,
                                                                                    y_val=y_val, y_test=y_test)
    seq_len: int = x_train.shape[1]
    input_dim: int = x_train.shape[2]
    output_dim: int = x_train.shape[2]
    # print(x_train.shape, x_val.shape, x_test.shape, y_train.shape, y_val.shape, y_test.shape)

    net = AgentGRU(seq_len=seq_len, input_size=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
                   num_layers=num_layers)
    optimizer = torch.optim.Adam(params=net.parameters(), lr=lr, weight_decay=1e-3)
    loss_fn = nn.BCELoss()

    print(net)
    params: Dict[str, Any] = {'epochs': NUM_EPOCHS, 'hidden_dim': hidden_dim, 'input_dim': input_dim,
                              'num_layers': num_layers,
                              'Number of Features': x_train.shape[-1]}
    net = train(net=net, train_dataloader=train_dataloader, val_dataloader=val_dataloader,
                test_dataloader=test_dataloader)
    print(f'Final Test Loss: {infer(net=net, infer_dataloader=test_dataloader, loss_fn=loss_fn):.2f}')
