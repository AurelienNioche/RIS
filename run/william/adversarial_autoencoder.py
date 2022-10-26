import os
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
from scipy import signal
import itertools


class Dataset(torch.utils.data.Dataset):
    """Characterizes a dataset for PyTorch"""

    def __init__(self, x):
        """Initialization"""
        self.x = torch.from_numpy(x.copy()).float()

    def __len__(self):
        """Denotes the total number of samples"""
        return self.x.shape[0]

    def __getitem__(self, index):
        """Generates one sample of data"""
        # Select sample
        x = self.x[index]
        return x, torch.zeros_like(x)

    @classmethod
    def load_datasets(cls, data_file):
        df = pd.read_csv(data_file, index_col=0)

        idx_data = df.columns[1:]

        labels = list(df.label.unique())

        training_data = {}
        for label in labels:
            x = df[df.label == label][idx_data].values
            x = signal.decimate(x, 4, axis=1)
            training_data[label] = cls(x)
        return training_data


def sample_normal(mu, logvar, latent_dim):
    std = torch.exp(0.5 * logvar)
    shape = (mu.size(0), latent_dim)
    rn = torch.randn(shape)
    z = rn * std + mu
    return z


class SavableModel(nn.Module):

    def __init__(self):
        super(SavableModel, self).__init__()
        self.construction_kwargs = {}

    @classmethod
    def base_name(cls, folder):
        return f"{folder}/{cls.__name__.lower()}"

    @classmethod
    def paths(cls, folder):
        return {
            "construction_kwargs": f"{cls.base_name(folder)}_construction_kwargs.torch",
            "state_dict": f"{cls.base_name(folder)}_state_dict.torch"
        }

    def save(self, folder):
        os.makedirs(folder, exist_ok=True)
        paths = self.paths(folder)
        torch.save(self.construction_kwargs, paths["construction_kwargs"])
        torch.save(self.state_dict(), paths["state_dict"])

    @classmethod
    def load(cls, folder):
        paths = cls.paths(folder)
        model = cls(**torch.load(paths["construction_kwargs"]))
        model.load_state_dict(torch.load(paths["state_dict"]))
        model.eval()
        return model


class Encoder(SavableModel):
    def __init__(self, input_dim, latent_dim):
        super(Encoder, self).__init__()

        self.construction_kwargs = dict(latent_dim=latent_dim,
                                        input_dim=input_dim)

        self.latent_dim = latent_dim

        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True))

        self.mu = nn.Linear(512, latent_dim)
        self.logvar = nn.Linear(512, latent_dim)

    def forward(self, x):
        x_ = self.model(x)
        mu = self.mu(x_)
        logvar = self.logvar(x_)
        z = sample_normal(mu, logvar, self.latent_dim)
        return z


class Decoder(SavableModel):
    def __init__(self, latent_dim, input_dim):
        super(Decoder, self).__init__()

        self.construction_kwargs = dict(latent_dim=latent_dim,
                                        input_dim=input_dim)

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, input_dim),
            nn.Tanh())

    def forward(self, z):
        x = self.model(z)
        return x


class Discriminator(SavableModel):
    def __init__(self, latent_dim):
        super(Discriminator, self).__init__()

        self.construction_kwargs = dict(latent_dim=latent_dim)

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid())

    def forward(self, z):
        validity = self.model(z)
        return validity.squeeze()


def generate(n_sample, latent_dim, epoch, folder, decoder):
    """Saves a grid of generated curves"""

    # Sample noise
    z = torch.randn((n_sample, latent_dim))
    gen_x = decoder(z)

    fig, axes = plt.subplots(nrows=n_sample)
    for i in range(n_sample):
        ax = axes[i]
        ax.plot(gen_x.data[i, :])

    plt.tight_layout()

    os.makedirs(folder, exist_ok=True)
    plt.savefig(f"{folder}/{epoch}.png")
    plt.close()


def train(condition, dataset,
          latent_dim, n_epochs, lr, b1, b2, sample_interval,
          n_sample,
          fig_folder,
          bkp_folder):

    # Use binary cross-entropy loss
    adversarial_loss = torch.nn.BCELoss()
    reconstruction_loss = torch.nn.MSELoss()

    # Configure data loader
    batch_size = len(dataset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    input_dim = dataset.x.shape[1]

    # Initialize generator and discriminator
    encoder = Encoder(input_dim=input_dim, latent_dim=latent_dim)
    decoder = Decoder(input_dim=input_dim, latent_dim=latent_dim)
    discriminator = Discriminator(latent_dim=latent_dim)

    opt_gen = torch.optim.Adam(
        itertools.chain(encoder.parameters(), decoder.parameters()), lr=lr, betas=(b1, b2))

    opt_dsc = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

    hist_loss = {k: [] for k in ("generation", "discrimination")}

    with tqdm(total=n_epochs, desc=condition) as pbar:
        for epoch in range(n_epochs):

            for _, (x, _) in enumerate(dataloader):

                # -----------------
                #  Train Generator
                # -----------------

                opt_gen.zero_grad()

                encoded = encoder(x)
                decoded = decoder(encoded)
                p_drawn_from_prior_dist = discriminator(encoded)

                n = x.shape[0]

                # Adversarial ground truths
                yes = torch.ones(n)
                no = torch.zeros(n)

                adv_loss = adversarial_loss(input=p_drawn_from_prior_dist, target=yes)
                rec_loss = reconstruction_loss(input=decoded, target=x)

                # Loss measures generator's ability to fool the discriminator
                g_loss = 0.001 * adv_loss + 0.999 * rec_loss

                g_loss.backward()
                opt_gen.step()

                # ---------------------
                #  Train Discriminator
                # ---------------------

                opt_dsc.zero_grad()

                # Sample noise as discriminator ground truth
                z = torch.randn((n, latent_dim))
                z__p_drawn_from_prior = discriminator(z)
                p_drawn_from_prior_dist = discriminator(encoded.detach())

                # Measure discriminator's ability to classify real from generated samples
                d_loss = 0.5 * adversarial_loss(input=p_drawn_from_prior_dist, target=no)
                d_loss += 0.5 * adversarial_loss(input=z__p_drawn_from_prior, target=yes)

                d_loss.backward()
                opt_dsc.step()

                hist_loss["generation"].append(g_loss.item())
                hist_loss["discrimination"].append(d_loss.item())

            if epoch > 0 and epoch % sample_interval == 0:
                generate(n_sample=n_sample, epoch=epoch,
                         decoder=decoder,
                         latent_dim=latent_dim,
                         folder=f"{fig_folder}/generated/{condition}/")

            pbar.set_postfix(
                loss_gen=g_loss.item(),
                loss_dsc=d_loss.item())
            pbar.update()

    for k, loss in hist_loss.items():
        folder = f"{fig_folder}/loss/{condition}"
        os.makedirs(folder, exist_ok=True)
        fig, ax = plt.subplots()
        ax.set_title(f"Loss {k}")
        ax.plot(loss)
        plt.tight_layout()
        plt.savefig(f"{folder}/loss_{k}.pdf")
        plt.close()

    models = [encoder, decoder, discriminator]
    for m in models:
        m.save(folder=f"{bkp_folder}/{condition}")


def main():

    root_folder = "../.."
    data_file = f"{root_folder}/data/william/preprocessed_data.csv"
    fig_folder = f"{root_folder}/fig/william"
    bkp_folder = f"{root_folder}/bkp/william/generative_models"

    for folder in fig_folder, bkp_folder:
        os.makedirs(folder, exist_ok=True)

    # For autoencoder
    latent_dim = 3

    # Optimizers
    lr = 0.005
    b1 = 0.3
    b2 = 0.999

    n_epochs = 1000

    sample_interval = 50
    n_sample = 5

    datasets = Dataset.load_datasets(data_file=data_file)

    for condition, dataset in datasets.items():
        train(
            condition=condition,  # "sitting", "standing"
            dataset=dataset,
            n_epochs=n_epochs,
            lr=lr, b1=b1, b2=b2, latent_dim=latent_dim,
            sample_interval=sample_interval, n_sample=n_sample,
            bkp_folder=bkp_folder,
            fig_folder=fig_folder)


if __name__ == "__main__":

    main()
