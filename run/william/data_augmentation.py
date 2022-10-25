import os

import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from scipy import signal
import itertools


class Dataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, x):
        'Initialization'
        self.x = torch.from_numpy(x.copy()).float()

    def __len__(self):
        'Denotes the total number of samples'
        return self.x.shape[0]

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        x = self.x[index]
        return x, torch.zeros_like(x)


def get_data(data_folder):
    f_name = f'{data_folder}/preprocessed_data.csv'
    df = pd.read_csv(f_name, index_col=0)

    idx_data = df.columns[1:]

    labels = list(df.label.unique())
    print(f"N labels = {len(labels)}")

    training_data = {}
    for label in labels:
        x = df[df.label == label][idx_data].values
        x = signal.decimate(x, 4, axis=1)
        training_data[label] = Dataset(x)
    return training_data


def sample_normal(mu, logvar, latent_dim):
    std = torch.exp(0.5 * logvar)
    shape = (mu.size(0), latent_dim)
    rn = torch.randn(shape)
    z = rn * std + mu
    return z


class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Encoder, self).__init__()

        self.latent_dim = latent_dim

        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.mu = nn.Linear(512, latent_dim)
        self.logvar = nn.Linear(512, latent_dim)

    def forward(self, x):
        x_ = self.model(x)
        mu = self.mu(x_)
        logvar = self.logvar(x_)
        z = sample_normal(mu, logvar, self.latent_dim)
        return z


class Decoder(nn.Module):
    def __init__(self, latent_dim, input_dim):
        super(Decoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, input_dim),
            nn.Tanh(),
        )

    def forward(self, z):
        x = self.model(z)
        return x


class Discriminator(nn.Module):
    def __init__(self, latent_dim):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        validity = self.model(z)
        return validity.squeeze()


def generate(n_row, latent_dim, batches_done, folder, decoder):
    """Saves a grid of generated digits"""
    # Sample noise
    z = torch.randn((n_row, latent_dim))
    gen_x = decoder(z)

    fig, axes = plt.subplots(nrows=n_row)
    for i in range(n_row):
        ax = axes[i]
        ax.plot(gen_x.data[i, :])

    plt.savefig(f"{folder}/{batches_done}.png")
    plt.close()


def train(train_data, latent_dim, gen_folder, n_epochs, lr, b1, b2, sample_interval):

    # Use binary cross-entropy loss
    adversarial_loss = torch.nn.BCELoss()
    reconstruction_loss = torch.nn.L1Loss()

    # Configure data loader
    batch_size = len(train_data)
    dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    input_dim = train_data.x.shape[1]

    # Initialize generator and discriminator
    encoder = Encoder(input_dim=input_dim, latent_dim=latent_dim)
    decoder = Decoder(input_dim=input_dim, latent_dim=latent_dim)
    discriminator = Discriminator(latent_dim=latent_dim)

    optimizer_G = torch.optim.Adam(
        itertools.chain(encoder.parameters(), decoder.parameters()), lr=lr, betas=(b1, b2))

    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

    for epoch in range(n_epochs):
        i = 0
        for i, (batch_x, _) in enumerate(dataloader):
            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            encoded = encoder(batch_x)
            decoded = decoder(encoded)
            predicted = discriminator(encoded)

            n = batch_x.shape[0]

            # Adversarial ground truths
            valid = torch.ones(n, requires_grad=False)
            fake = torch.zeros(n, requires_grad=False)

            adv_loss = adversarial_loss(input=predicted.squeeze(), target=valid)
            rec_loss = reconstruction_loss(input=decoded, target=batch_x)

            # Loss measures generator's ability to fool the discriminator
            g_loss = 0.001 * adv_loss + 0.999 * rec_loss

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Sample noise as discriminator ground truth
            z = torch.randn(batch_x.shape[0], latent_dim)

            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(discriminator(z), valid)
            fake_loss = adversarial_loss(discriminator(encoded.detach()), fake)
            d_loss = 0.5 * (real_loss + fake_loss)

            d_loss.backward()
            optimizer_D.step()

        batches_done = epoch * len(dataloader) + i

        if epoch > 10:
            if batches_done % sample_interval == 0:
                generate(n_row=5, batches_done=batches_done,
                         decoder=decoder,
                         latent_dim=latent_dim,
                         folder=gen_folder)

        if epoch % 5 == 0:
            print(
                f"[Epoch {epoch}/{n_epochs}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]"
            )


def main():

    root_folder = "../.."
    data_folder = f"{root_folder}/data/william"
    fig_folder = f"{root_folder}/fig/william"
    gen_folder = f"{fig_folder}/generated"

    for folder in fig_folder, data_folder, gen_folder:
        os.makedirs(folder, exist_ok=True)

    # For autoencoder
    latent_dim = 3

    # Optimizers
    lr = 0.005
    b1 = 0.3
    b2 = 0.999

    n_epochs = 1000

    sample_interval = 50

    train_data = get_data(data_folder=data_folder)
    for label, data in train_data.items():
        print(f"Using {label} data")
        train(gen_folder=gen_folder, train_data=data, n_epochs=n_epochs,
              lr=lr, b1=b1, b2=b2, latent_dim=latent_dim, sample_interval=sample_interval)


if __name__ == "__main__":

    main()
