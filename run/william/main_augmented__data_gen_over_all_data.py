from tqdm import tqdm
import torch
import numpy as np
from sklearn.model_selection import KFold
import pandas as pd
import os

from main import Net, Dataset, evaluate
from adversarial_autoencoder import Encoder, Decoder


def reset_weights(m):
    reset_parameters = getattr(m, "reset_parameters", None)
    if callable(reset_parameters):
        m.reset_parameters()


def train(data_file, gen_data_file, seed, k_folds, n_epochs,
          learning_rate):

    torch.manual_seed(seed)
    np.random.seed(seed)  # for sklearn

    dataset = Dataset(data_file=data_file)
    gen_dataset = Dataset(data_file=gen_data_file)

    kfold = KFold(n_splits=k_folds, shuffle=True)

    n_label = len(dataset.y.unique())
    model = Net(len_input=dataset.x.shape[-1],
                len_output=n_label)

    loss_fn = torch.nn.CrossEntropyLoss()

    folds_acc = {k: [] for k in ("training", "validation")}

    for fold, (train_idx, test_idx) in enumerate(kfold.split(dataset)):

        train_dataset = torch.utils.data.Subset(dataset, train_idx)
        test_dataset = torch.utils.data.Subset(dataset, test_idx)

        train_dataset = torch.utils.data.ConcatDataset([train_dataset, gen_dataset])

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=len(train_dataset), shuffle=True)
        val_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=len(test_dataset))

        model.apply(reset_weights)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

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

                acc_training = evaluate(model, train_loader).item()
                acc_validation = evaluate(model, val_loader).item()
                pbar.set_postfix(acc_training=acc_training, acc_validation=acc_validation)
                pbar.update()

            acc = evaluate(model, train_loader).item()
            folds_acc["training"].append(acc)

            acc = evaluate(model, val_loader).item()
            folds_acc["validation"].append(acc)
            pbar.set_postfix(acc_validation=acc)

    for k, v in folds_acc.items():
        print(f"{k.capitalize()} accuracy: {np.mean(v):.3f} (+/-{np.std(v):.3f})")


def main():

    data_file = "../../data/william/dataset2/preprocessed_data.csv"
    file_for_export = f'../../data/william/dataset2/generated_data.csv'
    bkp_folder = f"../../bkp/william/generative_models/dataset2"

    n = 10000

    k_folds = 10
    seed = 1234
    n_epochs = 200
    learning_rate = 0.01

    conditions = [x[0].split("/")[-1] for x in os.walk(bkp_folder)][1:]

    df_list = []

    for cond in conditions:
        folder = f"{bkp_folder}/{cond}"

        assert os.path.exists(folder), "Need to run `adversarial_autoencoder` first to generate models"

        encoder = Encoder.load(folder=folder)
        decoder = Decoder.load(folder=folder)
        z = torch.randn((n // 2, encoder.latent_dim))
        samples = decoder(z).detach().numpy()

        df = pd.DataFrame(samples)
        df.insert(0, 'label', cond)

        df_list.append(df)

    gen_data = pd.concat(df_list, axis=0, ignore_index=True, sort=False)
    gen_data.to_csv(file_for_export, index=True, header=True)

    train(data_file=data_file,
          seed=seed,
          k_folds=k_folds,
          n_epochs=n_epochs,
          gen_data_file=file_for_export,
          learning_rate=learning_rate)


if __name__ == "__main__":
    main()

