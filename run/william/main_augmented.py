from tqdm import tqdm
import torch
import numpy as np
from sklearn.model_selection import KFold

from main import Net, Dataset, evaluate
from adversarial_autoencoder import train as train_adversarial_auto_encoder


class GenDataset(torch.utils.data.Dataset):

    def __init__(self, train_dataset,
                 latent_dim,
                 lr_generation,
                 lr_discrimination,
                 n_epochs,
                 n_sample):
        """Initialization"""
        self.x, self.y = self.gen_data(
            train_dataset,
            latent_dim=latent_dim,
            lr_generation=lr_generation,
            lr_discrimination=lr_discrimination,
            n_epochs=n_epochs,
            n_sample=n_sample)

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.y)

    def __getitem__(self, index):
        """Generates one sample of data"""
        # Select sample
        x = self.x[index]
        y = self.y[index]
        return x, y

    @staticmethod
    def gen_data(
            train_dataset,
            latent_dim=10,
            lr_generation=0.001,
            lr_discrimination=0.00001,
            n_epochs=1000,
            n_sample=10000):

        samples_list = []
        labels_list = []

        dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset))
        for (x, y) in dataloader:

            uniq_labels = y.unique()

            for label in uniq_labels:

                # noinspection PyUnresolvedReferences
                idx = (y == label).nonzero()
                dataset = torch.utils.data.Subset(train_dataset, idx)

                encoder, decoder, discriminator = train_adversarial_auto_encoder(
                    dataset=dataset, latent_dim=latent_dim,
                    n_epochs=n_epochs, lr_generation=lr_generation,
                    lr_discrimination=lr_discrimination)

                z = torch.randn((n_sample//2, encoder.latent_dim))
                samples = decoder(z).detach()
                samples_list.append(samples)

                labels = torch.ones(n_sample//2) * label
                labels_list.append(labels)

        return torch.concat(samples_list), torch.concat(labels_list).long()


def reset_weights(m):
    reset_parameters = getattr(m, "reset_parameters", None)
    if callable(reset_parameters):
        m.reset_parameters()


def train(data_file,
          seed,
          k_folds,
          n_epochs,
          learning_rate,
          gen__latent_dim,
          gen__lr_generation,
          gen__lr_discrimination,
          gen__n_epochs,
          gen__n_sample):

    torch.manual_seed(seed)
    np.random.seed(seed)  # for sklearn

    dataset = Dataset(data_file=data_file)

    kfold = KFold(n_splits=k_folds, shuffle=True)

    n_label = len(dataset.y.unique())
    model = Net(len_input=dataset.x.shape[-1],
                len_output=n_label)

    loss_fn = torch.nn.CrossEntropyLoss()

    folds_acc = {k: [] for k in ("training", "validation")}

    for fold, (train_idx, test_idx) in enumerate(kfold.split(dataset)):

        train_dataset = torch.utils.data.Subset(dataset, train_idx)
        test_dataset = torch.utils.data.Subset(dataset, test_idx)

        gen_dataset = GenDataset(train_dataset,
                                 latent_dim=gen__latent_dim,
                                 lr_generation=gen__lr_generation,
                                 lr_discrimination=gen__lr_discrimination,
                                 n_epochs=gen__n_epochs,
                                 n_sample=gen__n_sample)

        train_dataset = torch.utils.data.ConcatDataset([train_dataset, gen_dataset])

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=len(train_dataset), shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=len(test_dataset))

        model.apply(reset_weights)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        # optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)

        with tqdm(total=n_epochs, desc=f"Fold {fold+1}/{k_folds}") as pbar:
            for _ in range(n_epochs):
                for i, data in enumerate(train_loader):
                    # Every data instance is an input + label pair
                    inputs, labels = data

                    # Zero your gradients for every batch!
                    optimizer.zero_grad()

                    # Make predictions for this batch
                    outputs = model(inputs)

                    # Compute the loss and its gradients
                    loss = loss_fn(outputs, labels)
                    loss.backward()

                    # Adjust learning weights
                    optimizer.step()

                acc_validation = evaluate(model, test_loader).item()
                acc_training = evaluate(model=model, dataloader=train_loader).item()
                pbar.set_postfix(acc_training=acc_training, acc_validation=acc_validation)
                pbar.update()

            acc_training = evaluate(model, train_loader).item()
            folds_acc["training"].append(acc_training)

            acc_validation = evaluate(model, test_loader).item()
            folds_acc["validation"].append(acc_validation)
            pbar.set_postfix(acc_training=acc_training, acc_validation=acc_validation)

    for k, v in folds_acc.items():
        print(f"{k.capitalize()} accuracy: {np.mean(v):.3f} (+/-{np.std(v):.3f})")


def main():

    data_file = "../../data/william/dataset2/preprocessed_data.csv"

    k_folds = 10
    seed = 123
    n_epochs = 1000
    learning_rate = 0.005

    gen__latent_dim = 10
    gen__lr_generation = 0.0005
    gen__lr_discrimination = 0.005
    gen__n_epochs = 1000
    gen__n_sample = 20000

    train(data_file=data_file,
          seed=seed,
          k_folds=k_folds,
          n_epochs=n_epochs,
          learning_rate=learning_rate,
          gen__latent_dim=gen__latent_dim,
          gen__lr_generation=gen__lr_generation,
          gen__lr_discrimination=gen__lr_discrimination,
          gen__n_epochs=gen__n_epochs,
          gen__n_sample=gen__n_sample)


if __name__ == "__main__":
    main()

