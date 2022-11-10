import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import os
import pywt


class Dataset(torch.utils.data.Dataset):
    """Characterizes a dataset for PyTorch"""

    def __init__(self, data_file,
                 wavelet='mexh',
                 scales=np.arange(1, 128, 2),
                 dt=1,
                 select_every=1,
                 decimate=None):

        """Initialization"""
        self.x, self.y = self.load_data(data_file=data_file, wavelet=wavelet,
                                        scales=scales,
                                        dt=dt,
                                        select_every=select_every,
                                        decimate=decimate)

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.y)

    def __getitem__(self, index):
        """Generates one sample of data"""
        x = self.x[index]
        y = self.y[index]
        return x, y

    @staticmethod
    def load_data(data_file, wavelet, scales, dt, decimate, select_every):

        df = pd.read_csv(data_file, index_col=0)
        data_columns = np.nonzero([c.isnumeric() for c in df.columns])[0]

        x = df.iloc[:, data_columns].values
        if decimate is not None:
            x = scipy.signal.decimate(x, axis=1, q=decimate)

        n, seq_length = x.shape

        df.label = pd.Categorical(df.label)
        y = df.label.cat.codes.values

        n_scale = len(scales)

        x2d = np.zeros((n, n_scale, seq_length))

        for i in range(n):
            [cfs, _] = pywt.cwt(x[i], scales, wavelet, dt)
            x2d[i, :, :] = np.abs(cfs)

        x2d = x2d[:, :, ::select_every]

        x = torch.from_numpy(x2d.copy()).float()
        x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
        y = torch.from_numpy(y.copy()).long()
        return x, y


class Net(nn.Module):
    def __init__(self, n_label):
        super().__init__()
        # self.nn = nn.Sequential(
        #     nn.Conv2d(1, 6, 5),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, 2),
        #     nn.Conv2d(6, 16, 5),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, 2),
        #     nn.Flatten(),
        #     nn.LazyLinear(120),
        #     nn.ReLU(),
        #     nn.Linear(120, 84),
        #     nn.ReLU(),
        #     nn.Linear(84, n_label)
        # )
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.LazyLinear(120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_label)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


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


def train(data_file,
          fig_folder=None,
          seed=0,
          wavelet='morl',
          dt=1,
          scales=np.arange(1, 128, 2),
          n_epochs=1000,
          learning_rate=0.005,
          verbose=True,
          decimate=5,
          select_every=1):

    torch.manual_seed(seed)

    data = Dataset(data_file=data_file, wavelet=wavelet, scales=scales, dt=dt,
                   decimate=decimate,
                   select_every=select_every)
    n_obs = len(data)

    n_training = int(0.8*n_obs)
    n_val = n_obs - n_training

    training_data, val_data = torch.utils.data.random_split(data, [n_training, n_val])

    train_loader = torch.utils.data.DataLoader(training_data,
                                               batch_size=len(training_data),
                                               shuffle=True)

    val_loader = torch.utils.data.DataLoader(val_data,
                                             batch_size=len(training_data),
                                             shuffle=True)

    n_label = len(data.y.unique())
    model = Net(n_label=n_label)

    loss_fn = torch.nn.CrossEntropyLoss()

    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Log a few stuff
    if verbose:
        print("n training", n_training)
        print("n val", n_val)
        acc = evaluate(model, train_loader)
        print(f"Accuracy before training on TRAINING = {acc}")
        acc = evaluate(model, val_loader)
        print(f"Accuracy before training on VALIDATION = {acc}")

    hist_loss = []
    hist_acc = []

    with tqdm(total=n_epochs) as pbar:
        for _ in range(n_epochs):
            for inputs, labels in train_loader:

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

            acc_training = evaluate(model, train_loader).item()
            acc_validation = evaluate(model, val_loader).item()
            pbar.set_postfix(acc_training=acc_training, acc_validation=acc_validation)
            pbar.update()

            hist_acc.append(acc_training)

    if fig_folder is not None:
        os.makedirs(fig_folder, exist_ok=True)
        fig, ax = plt.subplots()
        ax.set_title(f"Loss (CE)")
        ax.plot(hist_loss)
        plt.savefig(f"{fig_folder}/hist_loss.png")

        fig, ax = plt.subplots()
        ax.set_title(f"Accuracy")
        ax.plot(hist_acc)
        plt.savefig(f"{fig_folder}/hist_acc.png")

    acc_training = evaluate(model, train_loader)
    acc_validation = evaluate(model, val_loader)

    if verbose:
        print(f"Accuracy AFTER training on TRAINING = {acc_training}")
        print(f"Accuracy AFTER training on VALIDATION = {acc_validation}")

    return {"accuracy_training": acc_training, "accuracy_validation": acc_validation}


def main():

    wavelet = 'cgau1'  # 'morl'  # 'cmor' # 'cgau6'
    scales = np.geomspace(10, 520, num=20, dtype=int)
    dt = 1

    decimate = None
    select_every = 10

    data_file = f"../../data/william/dataset2/preprocessed_data__no_decimate.csv"
    fig_folder = "../../fig/william/main__wavelet_cnn/dataset2"

    train(
        data_file=data_file,
        fig_folder=fig_folder,
        learning_rate=0.005,
        seed=12,
        wavelet=wavelet,
        scales=scales,
        dt=dt,
        decimate=decimate,
        select_every=select_every,
        verbose=True)


if __name__ == "__main__":
    main()
