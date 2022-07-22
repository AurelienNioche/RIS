import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import os


CONDITIONS = conditions = "IRS-OFF-CorridorJunction", "IRS-OFF-Multifloor", "IRS-ON-CorridorJunction", "IRS-ON-Multifloor"


class Net(nn.Module):
    def __init__(self, len_input, len_ouput, hidden_layer_size=512):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(len_input, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, len_ouput),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


def get_files(condition, action, subject):

    if subject == 1:
        sbj = ""
    elif subject == 2:
        sbj = "S2"
    else:
        raise ValueError


    if condition == "IRS-OFF-CorridorJunction":
        if action == "Empty":
            fp = "IRS_OFF_Empty_Rx*.csv"
        elif action == "Sitting" and subject == 1:
            fp = "IRS_OF_Sitting_Tx*.csv"
        elif action == "Standing" and subject == 1:
            fp = "IRS_OF_Standing_Tx*.csv"
        else:
            fp = f"IRS_OFF_{action}{sbj}_Tx*.csv"

    elif condition == "IRS-OFF-Multifloor":
        if action == "Empty":
            fp = "EmptyOFF_*.csv"
        else:
            fp = f"{action}{sbj}_IRS-OFF_*.csv"

    elif condition == "IRS-ON-CorridorJunction":
        if action == "Empty":
            raise ValueError
        else:
            fp = f"IRS_ON_{action}{sbj}_Tx*.csv"

    elif condition == "IRS-ON-Multifloor":
        if action == "Empty":
            fp = "Empty_*.csv"
        else:
            fp = f"{action}{sbj}_IRS-ON_*.csv"

    else:
        raise ValueError


    files = glob.glob(f"data/{condition}/{fp}")
    assert len(files), f"No files found with pattern {fp} for condition {condition}"
    return files

def get_actions(condition):

    actions = ["Sitting", "Walking", "Standing", "Empty"]

    if condition == "IRS-ON-CorridorJunction":
        actions.remove("Empty")

    return actions


class Dataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, x, y):
        'Initialization'
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.y)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        x = self.x[index]
        y = self.y[index]
        return x, y


def get_data(condition, subject):

    actions = get_actions(condition)

    print(f"N labels = {len(actions)}")

    len_records = []

    for a in actions:
        for f in get_files(condition=condition, action=a, subject=subject):
            df_f = pd.read_csv(f, header=None)
            len_records.append(df_f.shape[-1])

    shortest_seq = min(len_records)
    print(f"Shortest sequence = {shortest_seq}")

    n_obs = len(len_records)
    print(f"Number of observations = {n_obs}")

    x = np.zeros((n_obs, shortest_seq))
    y = np.zeros(n_obs)

    i = 0
    for a in actions:
        for f in get_files(condition=condition, action=a, subject=subject):
            df_f = pd.read_csv(f, header=None)
            x[i] = df_f.mean()[:shortest_seq]
            y[i] = actions.index(a)
            i += 1

    training_data = Dataset(x, y)
    return training_data


def evaluate(model, dataloader):

    # initialize metric
    metric = torchmetrics.Accuracy()

    for i, data in enumerate(dataloader):
        # Every data instance is an input + label pair
        inputs, labels = data

        # Make predictions for this batch
        outputs = model(inputs)
        preds = outputs.softmax(dim=-1)

        # metric on current batch
        acc = metric(preds, labels)

    acc = metric.compute()

    return acc


def train(condition, subject):

    data = get_data(condition=condition, subject=subject)
    n_obs = len(data)

    n_training = int(0.80*n_obs)
    n_val = n_obs - n_training

    training_data, val_data = torch.utils.data.random_split(data, [n_training, n_val])

    train_dataloader = torch.utils.data.DataLoader(training_data,
                                                   batch_size=len(training_data),
                                                   shuffle=True)

    val_dataloader = torch.utils.data.DataLoader(val_data,
                                                 batch_size=len(training_data),
                                                 shuffle=True)

    model = Net(len_input=data.x.shape[-1],
                len_ouput=len(data.y.unique()))

    acc = evaluate(model, train_dataloader)
    print(f"[{condition} S{subject}] Accuracy before training on TRAINING = {acc}")

    acc = evaluate(model, val_dataloader)
    print(f"[{condition} S{subject}] Accuracy before training on VALIDATION = {acc}")

    loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    n_epochs = 10000
    hist_loss = []
    hist_acc = []

    # initialize metric
    metric = torchmetrics.Accuracy()

    for _ in tqdm(range(n_epochs)):
        for i, data in enumerate(train_dataloader):
            # Every data instance is an input + label pair
            inputs, labels = data

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

    os.makedirs("fig", exist_ok=True)
    fig, ax = plt.subplots()
    ax.set_title(f"{condition} - S{subject} - Loss (CE)")
    ax.plot(hist_loss)
    plt.savefig(f"fig/{condition}_S{subject}_hist_loss.pdf")

    fig, ax = plt.subplots()
    ax.set_title(f"{condition} - S{subject} - Accuracy")
    ax.plot(hist_acc)
    plt.savefig(f"fig/{condition}_S{subject}_hist_acc.pdf")

    acc = evaluate(model, train_dataloader)
    print(f"[{condition} S{subject}] Accuracy AFTER training on TRAINING = {acc}")

    acc = evaluate(model, val_dataloader)
    print(f"[{condition} S{subject}] Accuracy AFTER training on VALIDATION = {acc}")


def main():
    conditions = "IRS-OFF-CorridorJunction", "IRS-OFF-Multifloor", "IRS-ON-CorridorJunction", "IRS-ON-Multifloor"

    for condition in conditions:
        for subject in (1, 2):
            train(condition=condition, subject=subject)

if __name__ == "__main__":
    main()

