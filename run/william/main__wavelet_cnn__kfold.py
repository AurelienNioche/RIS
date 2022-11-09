import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import numpy as np
from sklearn.model_selection import KFold

from main__wavelet_cnn import Net, Dataset, evaluate


def reset_weights(m):
    reset_parameters = getattr(m, "reset_parameters", None)
    if callable(reset_parameters):
        m.reset_parameters()


def train(data_file, fig_folder, seed, k_folds, n_epochs,
          learning_rate,
          preprocess_kwargs):

    torch.manual_seed(seed)
    np.random.seed(seed)  # for sklearn

    dataset = Dataset(data_file=data_file, **preprocess_kwargs)

    kfold = KFold(n_splits=k_folds, shuffle=True)

    n_label = len(dataset.y.unique())
    model = Net(n_label=n_label)

    loss_fn = torch.nn.CrossEntropyLoss()

    folds_acc = {k: [] for k in ("training", "validation")}

    for fold, (train_idx, test_idx) in enumerate(kfold.split(dataset)):

        train_dataset = torch.utils.data.Subset(dataset, train_idx)
        test_dataset = torch.utils.data.Subset(dataset, test_idx)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=len(train_idx))
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=len(test_idx))

        model.apply(reset_weights)
        # optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

        hist_acc = []
        hist_loss = []

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
                acc_validation = evaluate(model, test_loader).item()
                pbar.set_postfix(acc_training=acc_training, acc_validation=acc_validation)
                pbar.update()

                hist_loss.append(loss.item())
                hist_acc.append(acc_training)

            train_acc = evaluate(model, train_loader).item()
            val_acc = evaluate(model, test_loader).item()
            folds_acc["validation"].append(val_acc)
            folds_acc["training"].append(train_acc)
            pbar.set_postfix(acc_training=train_acc, acc_validation=val_acc)

            fig_folder_fold = f"{fig_folder}/fold{fold+1}"
            os.makedirs(fig_folder_fold, exist_ok=True)
            fig, ax = plt.subplots()
            ax.set_title(f"Loss (CE)")
            ax.plot(hist_loss)
            plt.savefig(f"{fig_folder_fold}/hist_loss.png")
            plt.close()

            fig, ax = plt.subplots()
            ax.set_title(f"Accuracy")
            ax.plot(hist_acc)
            plt.savefig(f"{fig_folder_fold}/hist_acc.png")
            plt.close()

    for k, v in folds_acc.items():
        print(f"{k.capitalize()} accuracy: {np.mean(v):.3f} (+/-{np.std(v):.3f})")


def main():

    data_file = "../../data/william/dataset2/preprocessed_data__no_decimate.csv"
    fig_folder = "../../fig/william/k_fold__wavelet_cnn/dataset2"

    k_folds = 10
    seed = 123
    n_epochs = 300
    learning_rate = 0.005
    preprocess_kwargs = dict(
        wavelet="cgau8",
        scales=np.geomspace(10, 520, num=20, dtype=int),
        dt=1,
        decimate=5,
        select_every=5)

    train(
        preprocess_kwargs=preprocess_kwargs,
        data_file=data_file,
        fig_folder=fig_folder,
        seed=seed,
        k_folds=k_folds,
        n_epochs=n_epochs,
        learning_rate=learning_rate)


if __name__ == "__main__":
    main()

