import torch
from torch import nn
from torch.nn.modules import loss
from torch.utils.data import DataLoader
from typing import Tuple, Dict, Any
import numpy as np
from tqdm import tqdm
from utils import load_hdf5_file, plot_values_by_epochs, check_earlystopping, calculate_metrics, load_X_y_from_disk, \
    _split_to_train_validation_test, get_dataloader_for_datasets

from consts import NUM_EPOCHS, train_ratio, validation_ratio, test_ratio, hidden_dim, num_layers, lr, batch_size, PRINT_EVERY, SAVE_EVERY

train_on_gpu = torch.cuda.is_available()


def save_pt_model(net: nn.Module) -> None:
    torch.save(net.state_dict(), 'Models/GRU_weights.pt')


def load_pt_model(input_size: int, hidden_dim: int, output_dim: int, num_layers: int = 2,
                  model_name: str = 'GRU_weights') -> nn.Module:
    net = AgentGRU(input_size=input_size, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
    net.load_state_dict(torch.load(f'Models/{model_name}.pt'))
    return net


class AgentGRU(nn.Module):
    def __init__(self, input_size: int, hidden_dim: int, output_dim: int, num_layers: int = 2,
                 dropout_prob: float = 0.5):
        super(AgentGRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: np.array, hidden: bool = None):
        """
        Forward pass of the net
        """
        if not hidden:
            h = self.init_params(x.size(0))
        else:
            h = hidden
        # print(f'x: {x.shape}, h: {h.shape}')
        out, hidden = self.gru(x, h.detach())
        # out = self.dropout(out)
        out = self.fc(out[:, -1, :])  # Fully Connected on the last timestamp
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
          test_dataloader: DataLoader = None, is_earlystopping: bool = True) -> nn.Module:
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
    best_epoch: int = NUM_EPOCHS - 1

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

            train_loss = loss_fn(torch.clamp(y_train_pred, min=0, max=1), y_train)
            train_losses[epoch] = train_loss.item() / len(train_dataloader)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            if val_dataloader:
                val_losses[epoch] = infer(net, val_dataloader, loss_fn)

            if test_dataloader:
                test_losses[epoch] = infer(net, test_dataloader, loss_fn)

            if is_earlystopping and check_earlystopping(loss=val_losses, epoch=epoch):
                print('EarlyStopping !!!')
                best_epoch = np.argmin(val_losses[:epoch + 1])
                break

        if epoch % PRINT_EVERY == 0:
            print(f"Epoch: {epoch + 1}/{NUM_EPOCHS},",
                  f"Train loss: {train_losses[epoch]:.5f},",
                  f"Validation loss: {val_losses[epoch]:.5f}")
            print(f"Test Loss: {test_losses[epoch]:.5f}" if test_dataloader else '')

        if epoch % SAVE_EVERY == 0:
            save_pt_model(net=net)

    if best_epoch != NUM_EPOCHS - 1:  # earlystopping NOT activated
        train_losses = train_losses[:best_epoch + 1]
        val_losses = val_losses[:best_epoch + 1]
        test_losses = test_losses[:best_epoch + 1]
    else:
        best_epoch = np.argmin(val_losses)
    # accuracy, recall, precision, f1score = calculate_metrics()
    print(
        f'Best Epoch: {best_epoch}; Best Validation Loss: {val_losses[best_epoch]:.4f} -> Test Loss: {test_losses[best_epoch]:.4f}')
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
    for x, y in infer_dataloader:
        with torch.no_grad():
            if train_on_gpu:
                x, y = x.cuda(), y.cuda()
            y_pred, hidden = net(x)
            infer_loss = loss_fn(input=torch.clamp(y_pred.reshape(-1), min=0, max=1), target=y.reshape(-1))
        running_loss += infer_loss
    return running_loss / len(infer_dataloader)


if __name__ == '__main__':
    """
    # TODO - Usage of focal loss - https://discuss.pytorch.org/t/is-this-a-correct-implementation-for-focal-loss-in-pytorch/43327/6
    # TODO - Try to use bidirectional model
    """
    X, y = load_X_y_from_disk(num_features=1)
    print(X.shape)
    x_train, x_val, x_test, y_train, y_val, y_test = _split_to_train_validation_test(X=X, y=y,
                                                                                     train_ratio=train_ratio,
                                                                                     validation_ratio=validation_ratio)

    train_dataloader, val_dataloader, test_dataloader = get_dataloader_for_datasets(x_train=x_train, x_val=x_val,
                                                                                    x_test=x_test, y_train=y_train,
                                                                                    y_val=y_val, y_test=y_test)
    seq_len: int = x_train.shape[1]
    input_dim: int = x_train.shape[2]
    output_dim: int = x_train.shape[2]

    net = AgentGRU(input_size=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
    optimizer = torch.optim.Adam(params=net.parameters(), lr=lr, weight_decay=1e-3)
    loss_fn = nn.BCELoss()
    # print('Starting After 11 Epoches !')
    # net = load_pt_model(input_size=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
    print(net)
    net = train(net=net, train_dataloader=train_dataloader, val_dataloader=val_dataloader,
                test_dataloader=test_dataloader, is_earlystopping=True)
    print(f'Final Test Loss: {infer(net=net, infer_dataloader=test_dataloader, loss_fn=loss_fn):.2f}')


    save_pt_model(net=net)
