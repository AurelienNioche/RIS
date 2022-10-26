from tqdm import tqdm
import torch
import numpy as np
from sklearn.model_selection import KFold

from main import Net, Dataset, evaluate


def reset_weights(m):
    reset_parameters = getattr(m, "reset_parameters", None)
    if callable(reset_parameters):
        m.reset_parameters()


def train(data_file, seed, k_folds, n_epochs):

    torch.manual_seed(seed)

    dataset = Dataset(data_file=data_file)

    kfold = KFold(n_splits=k_folds, shuffle=True)

    n_label = len(dataset.y.unique())
    model = Net(len_input=dataset.x.shape[-1],
                len_output=n_label)

    loss_fn = torch.nn.CrossEntropyLoss()

    folds_acc = {k: [] for k in ("training", "validation")}

    for fold, (train_idx, test_idx) in enumerate(kfold.split(dataset)):

        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_idx)

        train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=len(train_idx), sampler=train_subsampler)
        test_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=len(test_idx), sampler=test_subsampler)

        model.apply(reset_weights)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

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

                acc = evaluate(model=model, dataloader=train_loader).item()
                pbar.set_postfix(acc=acc)
                pbar.update()

        acc = evaluate(model, train_loader).item()
        # print(f"Accuracy on TRAINING = {acc:.3f}")
        folds_acc["training"].append(acc)

        acc = evaluate(model, test_loader).item()
        # print(f"Accuracy on VALIDATION = {acc:.3f}")
        folds_acc["validation"].append(acc)

    for k, v in folds_acc.items():
        print(f"{k.capitalize()} accuracy: {np.mean(v):.3f} (+/-{np.std(v):.3f})")


def main():

    data_file = "../../data/william/preprocessed_data.csv"

    k_folds = 10
    seed = 1234
    n_epochs = 1000

    train(data_file=data_file,
          seed=seed,
          k_folds=k_folds,
          n_epochs=n_epochs)


if __name__ == "__main__":
    main()

