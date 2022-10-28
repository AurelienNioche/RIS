import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torchmetrics
import os


class Dataset(torch.utils.data.Dataset):
    """Characterizes a dataset for PyTorch"""

    def __init__(self, data_file):
        """Initialization"""
        self.x, self.y = self.load_data(data_file)

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.y)

    def __getitem__(self, index):
        """Generates one sample of data"""
        x = self.x[index]
        y = self.y[index]
        return x, y

    @staticmethod
    def load_data(data_file):

        df = pd.read_csv(data_file, index_col=0)

        x = df.iloc[:, 1:].values

        df.label = pd.Categorical(df.label)
        y = df.label.cat.codes.values

        x = torch.from_numpy(x.copy()).float()
        y = torch.from_numpy(y.copy()).long()
        return x, y


class Net(nn.Module):
    def __init__(self, len_input, len_output, hidden_layer_size=512):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(len_input, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, len_output))

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


def evaluate(model, dataloader):

    # initialize metric
    metric = torchmetrics.Accuracy()

    with torch.no_grad():
        for inputs, labels in dataloader:

            outputs = model(inputs)
            pred = outputs.softmax(dim=-1)
            _ = metric(pred, labels)

        acc = metric.compute()

    return acc


def train(data_file, fig_folder, seed):

    torch.manual_seed(seed)

    data = Dataset(data_file=data_file)
    n_obs = len(data)

    n_training = int(0.8*n_obs)
    n_val = n_obs - n_training

    print("n training", n_training)
    print("n val", n_val)

    training_data, val_data = torch.utils.data.random_split(data, [n_training, n_val])

    train_dataloader = torch.utils.data.DataLoader(training_data,
                                                   batch_size=len(training_data),
                                                   shuffle=True)

    val_dataloader = torch.utils.data.DataLoader(val_data,
                                                 batch_size=len(training_data),
                                                 shuffle=True)

    n_label = len(data.y.unique())
    model = Net(len_input=data.x.shape[-1],
                len_output=n_label)

    acc = evaluate(model, train_dataloader)
    print(f"Accuracy before training on TRAINING = {acc}")

    acc = evaluate(model, val_dataloader)
    print(f"Accuracy before training on VALIDATION = {acc}")

    loss_fn = torch.nn.CrossEntropyLoss()

    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    n_epochs = 1000
    hist_loss = []
    hist_acc = []

    # initialize metric
    metric = torchmetrics.Accuracy()

    with tqdm(total=n_epochs) as pbar:
        for _ in range(n_epochs):
            for inputs, labels in train_dataloader:

                # Zero your gradients for every batch!
                optimizer.zero_grad()

                # Make predictions for this batch
                outputs = model(inputs)

                # Compute the loss and its gradients
                loss = loss_fn(outputs, labels)
                loss.backward()

                hist_loss.append(loss.item())

                # Adjust learning weights
                optimizer.step()

                # Compute metric on current batch
                preds = outputs.softmax(dim=-1)
                _ = metric(preds, labels)

            acc = metric.compute()
            hist_acc.append(acc.item())
            pbar.set_postfix(acc=acc.item())
            pbar.update()

    os.makedirs(fig_folder, exist_ok=True)
    fig, ax = plt.subplots()
    ax.set_title(f"Loss (CE)")
    ax.plot(hist_loss)
    plt.savefig(f"{fig_folder}/hist_loss.png")

    fig, ax = plt.subplots()
    ax.set_title(f"Accuracy")
    ax.plot(hist_acc)
    plt.savefig(f"{fig_folder}/hist_acc.png")

    acc = evaluate(model, train_dataloader)
    print(f"Accuracy AFTER training on TRAINING = {acc}")

    acc = evaluate(model, val_dataloader)
    print(f"Accuracy AFTER training on VALIDATION = {acc}")


def main():

    data_file = "../../data/william/preprocessed_data.csv"
    fig_folder = "../../fig/william/main"

    seed = 12

    train(data_file=data_file,
          fig_folder=fig_folder,
          seed=seed)


if __name__ == "__main__":
    main()
