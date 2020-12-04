import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from typing import Tuple, Dict, Any
import numpy as np
from utils import load_hdf5_file, PRINT_EVERY

train_on_gpu = torch.cuda.is_available()

class AgentGRU(nn.Module):
    """
    In my case each epoch == batch (All the training data passes through the network in one batch)
    The hidden state should be kept in each Batch (Epoch in my case)
    Index hidden state of last time step
      out.size() --> 100, 28, 100
      out[:, -1, :] --> 100, 100 --> just want last time step hidden states!
    TODO - Check if when evaluation the weights should be zeroed also?
    """

    def __init__(self, input_size, hidden_dim, output_dim=1, num_layers=2, dropout_prob=0.5):
        super(AgentGRU, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        # self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden=None):
        """
        Forward pass of the net
        param x: torch.Size([n_batches/num_of_series(20), seq_len(24), num_features(1)])
        param h: tensor
        """
        # 1 Tensors - (ht)
        if not hidden:
            h = self.init_params(x.size(0))
        else:
            h = hidden
        out, hidden = self.gru(x, h.detach())
        out = self.dropout(out)
        out = self.fc(out[:, -1, :])
        return out, hidden

    def init_params(self, batch_size: int) -> torch.Tensor:
        """
        Init the weights of the hidden states with random
        return matrix of zeros in size (self.num_layers, batch_size, self.hidden_dim)
        """
        if train_on_gpu:
            return torch.zeros(self.num_layers, batch_size, self.hidden_dim).requires_grad_().cuda()
        return torch.zeros(self.num_layers, batch_size, self.hidden_dim).requires_grad_()


def train(net: nn.Module, x_train: np.array, y_train: np.array, x_validation: np.array, y_validation: np.array,
          train_dataloader: DataLoader = None, val_dataloader: DataLoader = None, test_dataloader: DataLoader = None):
    train_losses: np.array = np.zeros(NUM_EPOCHS)
    test_losses: np.array = np.zeros(NUM_EPOCHS)
    val_losses: np.array = np.zeros(NUM_EPOCHS)
    hidden_list: list = []

    if train_on_gpu:
        model.cuda()  # Sends the parameters to the cuda device
        x_train = x_train.cuda()
        y_train = y_train.cuda()
    if test_dataloader:
        untrianed_test_loss = infer(net, test_dataloader, loss_fn)
    for epoch in range(NUM_EPOCHS):
        model.train()
        y_train_pred, hidden = model(x_train)
        # We don't want to propogate the whole network
        hidden = hidden.data
        hidden_list.append(hidden)

        train_loss = loss_fn(y_train_pred, y_train)
        train_losses[epoch] = train_loss.item()

        # Infer mode
        model.eval()

        if train_on_gpu:
            x_test, y_test = x_test.cuda(), y_test.cuda()
        y_test_pred, hidden = model(x_test)

        test_loss = loss_fn(y_test_pred, y_test)
        test_loss = test_loss.item()
        test_losses[epoch] = np.sqrt(test_loss)
        # Zeros out gradient, else they will accumulate between epochs
        optimizer.zero_grad()
        # Backward pass
        train_loss.backward()
        # Update parameters
        optimizer.step()

        if val_dataloader:
            val_losses[epoch] = infer(net, val_dataloader, loss_fn)
        if epoch % PRINT_EVERY == 0:
            print(f"Epoch: {epoch}/{NUM_EPOCHS},",
                  f"Train loss: {train_losses[epoch]:.4f},",
                  f"Validation loss: {val_losses[epoch]:.4f}")


def infer(net: nn.Module, x_infer: np.array, y_infer: np.array):
    pass


class TargetsDataset(Dataset):
    """
    The dataset object used to read the data
    """
    def __init__(self, features:np.array, labels:np.array):
        self.features = featyres

    def __getitem__(self, idx):
        pass

    def __len__(self):
        pass


def load_X_y_from_disk() -> Tuple[np.array, np.array]:
    X, y = load_hdf5_file('X'), load_hdf5_file('y')
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


if __name__ == '__main__':
    """
    Usage of focal loss - https://discuss.pytorch.org/t/is-this-a-correct-implementation-for-focal-loss-in-pytorch/43327/6
    """
    NUM_EPOCHS: int = 50
    train_ratio: float = 0.7
    validation_ratio: float = 0.15
    test_ratio: float = 0.15
    X, y = load_X_y_from_disk()
    x_train, x_val, x_test, y_train, y_val, y_test = _split_to_train_validation_test(X=X, y=y,
                                                                                     train_ratio=train_ratio,
                                                                                     validation_ratio=validation_ratio)
    print(x_train.shape, x_val.shape, x_test.shape, y_train.shape, y_val.shape, y_test.shape)
    input_dim: int = x_train.shape[2]
    output_dim: int = x_train.shape[2]
    hidden_dim: int = 32
    num_layers: int = 2
    lr: float = 1e-3
    batch_size: int = 256

    model = AgentGRU(input_size=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=1e-3)
    loss_fn = nn.BCELoss()

    print(model)
    params: Dict[str, Any] = {'epochs': NUM_EPOCHS, 'hidden_dim': hidden_dim, 'input_dim': input_dim, 'num_layers': num_layers,
              'Number of Features': x_train.shape[-1]}
