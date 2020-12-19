import torch
from torch import nn
from torch.nn.modules import loss
from torch.utils.data import DataLoader
from typing import Tuple
import numpy as np
from tqdm import tqdm
from utils import load_hdf5_file, plot_values_by_epochs, check_earlystopping, load_X_y_from_disk, \
    split_to_train_validation_test, get_dataloader_for_datasets, calculate_model_metrics, \
    get_num_of_areas_and_targets_from_arary, load_data, save_pickle_object, print_data_statistics, print_model_parameters_count

from consts import NUM_EPOCHS, TRAIN_RATIO, VALIDATION_RATIO, TEST_RATIO, hidden_dim, num_layers, lr, BATCH_SIZE, \
    PRINT_EVERY, SAVE_EVERY

train_on_gpu = torch.cuda.is_available()


def save_pt_model(net: nn.Module) -> None:
    torch.save(net.state_dict(), 'Models/GRU_weights.pt')


def load_pt_model(input_size: int, hidden_dim: int, output_dim: int, num_layers: int = 2,
                  model_name: str = 'GRU_weights') -> nn.Module:
    net = AgentGRU(input_size=input_size, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
    net.load_state_dict(torch.load(f'Models/{model_name}.pt'))
    return net


class AgentGRU(nn.Module):
    def __init__(self, input_size: int, hidden_dim: int, output_dim: int, num_layers: int):
        super(AgentGRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size=input_size, hidden_size=self.hidden_dim, num_layers=self.num_layers,
                          batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: np.array, h: np.array = None):
        if h is None:
            h = self.init_hidden(x.shape[0])
        out, h = self.gru(x, h)
        out = self.sigmoid(self.fc(out[:, -1, :]))  # Fully Connected on the last timestamp
        return out, h

    def init_hidden(self, batch_size: int) -> torch.Tensor:
        """
        Init the weights of the hidden states with random
        return matrix of zeros in size (self.num_layers, batch_size, self.hidden_dim)
        """
        weight = next(self.parameters()).data
        hidden = weight.new(self.num_layers, batch_size, self.hidden_dim).zero_()
        if train_on_gpu:
            hidden = hidden.cuda()
        return hidden


def train(net: nn.Module, train_dataloader: DataLoader = None, val_dataloader: DataLoader = None,
          test_dataloader: DataLoader = None, is_earlystopping: bool = True) -> nn.Module:
    """
    Training loop iterating on the train dataloader and updating the model's weights.
    Inferring the validation dataloader & test dataloader, if given, to babysit the learning
    Activating cuda device if available.
    :return: Trained model
    """
    train_losses: np.array = np.zeros(NUM_EPOCHS)
    val_losses: np.array = np.zeros(NUM_EPOCHS)
    best_epoch: int = NUM_EPOCHS - 1

    if test_dataloader:
        untrained_test_loss, untrained_y_test_pred = infer(net, test_dataloader, loss_fn)
        _, _ = get_num_of_areas_and_targets_from_arary(array=y_test)
        print(f'Test Loss before training: {untrained_test_loss:.3f}')
        _, _, _ = calculate_model_metrics(y_true=y_test, y_pred=untrained_y_test_pred, verbose=True)

    for epoch in range(NUM_EPOCHS):
        print(f'*************** Epoch {epoch + 1} ***************')
        net.train()
        h = net.init_hidden(batch_size=BATCH_SIZE)
        for batch_idx, (x_train, y_train) in enumerate(tqdm(train_dataloader)):
            if train_on_gpu:
                net.cuda()
                x_train, y_train = x_train.cuda(), y_train.cuda()
            h = h.data
            optimizer.zero_grad()
            y_train_pred, h = net(x_train, h)
            loss = loss_fn(y_train_pred, y_train)
            loss.backward()
            optimizer.step()

        if val_dataloader:
            val_loss, y_val_pred = infer(net, val_dataloader, loss_fn)
            val_losses[epoch] = val_loss

        if is_earlystopping and check_earlystopping(loss=val_losses, epoch=epoch):
            print('EarlyStopping !!!')
            best_epoch = np.argmin(val_losses[:epoch + 1])
            break
        train_losses[epoch] = loss.item() / len(train_dataloader)
        scheduler.step(val_loss) # Change the lr if needed based on the validation loss

        if epoch % PRINT_EVERY == 0:
            print(f"Epoch: {epoch + 1}/{NUM_EPOCHS},",
                  f"Train loss: {train_losses[epoch]:.5f},",
                  f"Validation loss: {val_losses[epoch]:.5f}")

            _, _, _ = calculate_model_metrics(y_true=y_train, y_pred=y_train_pred, mode='Train-Last Batch')
            if val_dataloader:
                _, _, _ = calculate_model_metrics(y_true=y_val, y_pred=y_val_pred, mode='Validation')

        if (epoch + 1) % SAVE_EVERY == 0:
            save_pt_model(net=net)

    if best_epoch != NUM_EPOCHS - 1:  # earlystopping NOT activated
        train_losses = train_losses[:best_epoch + 1]
        val_losses = val_losses[:best_epoch + 1]
    else:
        best_epoch = np.argmin(val_losses)

    print(
        f'Best Epoch: {best_epoch + 1}; Best Validation Loss: {val_losses[best_epoch]:.4f}')
    print(train_losses)
    plot_values_by_epochs(train_values=train_losses, validation_values=val_losses)
    return net


def infer(net: nn.Module, infer_dataloader: DataLoader, loss_fn: loss) -> Tuple[float, np.array]:
    """
    Run the model on x_infer (both validation and test) and calculate the loss of the predictions.
    The model run on evaluation mode and without updating the computational graph (no_grad)
    Running on the dataloader by batches, defined in each dataset's DataLoader
    :return loss, y_pred
    """
    net.eval()
    running_loss = 0
    y_infer_pred = []
    for x, y in infer_dataloader:
        with torch.no_grad():
            if train_on_gpu:
                x, y = x.cuda(), y.cuda()
            y_pred, _ = net(x)
            infer_loss = loss_fn(input=y_pred.reshape(-1), target=y.reshape(-1))
            y_infer_pred.append(y_pred)
        running_loss += infer_loss
    return running_loss / len(infer_dataloader), torch.cat(tuple(y_infer_pred))



if __name__ == '__main__':
    """
    # TODO - Try to use bidirectional model
    """
    TRAINING_MODE = True
    print(
        '**************** Train Mode ****************' if TRAINING_MODE else '**************** Test Mode ****************')

    x_train, x_val, x_test, y_train, y_val, y_test = load_data()
    print_data_statistics(x_train, x_val, x_test, y_train, y_val, y_test)

    train_dataloader, val_dataloader, test_dataloader = get_dataloader_for_datasets(x_train=x_train, x_val=x_val,
                                                                                    x_test=x_test, y_train=y_train,
                                                                                    y_val=y_val, y_test=y_test)
    seq_len: int = x_train.shape[1]
    input_dim: int = x_train.shape[2]
    output_dim: int = y_train.shape[1]

    net = AgentGRU(input_size=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
    optimizer = torch.optim.Adam(params=net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min') # Reduce lr while the loss is stuck
    loss_fn = nn.BCELoss()

    if TRAINING_MODE:
        net = train(net=net, train_dataloader=train_dataloader, val_dataloader=val_dataloader, is_earlystopping=True)
        save_pt_model(net=net)
    else:
        print('Loading trained model')
        net = load_pt_model(input_size=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
        print(net)

    print_model_parameters_count(net)
    test_loss, y_test_pred = infer(net=net, infer_dataloader=test_dataloader, loss_fn=loss_fn)
    print(f'Final Test Loss: {test_loss:.2f}')
    print(f'Number of Epochs: {NUM_EPOCHS}')
    calculate_model_metrics(y_true=y_test, y_pred=y_test_pred)
