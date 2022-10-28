import pandas as pd
import numpy as np
import glob
from scipy import signal
from tqdm import tqdm


def preprocessing(data):

    # Remove First Subcarrier
    data = data.iloc[1:]
    
    # Flip sencond half of the subcarriers
    data_low = data.iloc[0:27]
    data_high = data.iloc[27:54] * -1
    data = pd.concat([data_low, data_high])

    # Take the average of all the subcarriers
    data = data.mean(axis=0)
    data = data.to_frame() 
    data = data.T 

    # Take the average of value all 2000 rows then subtract
    # the average from each column to scale the variations from zero
    avg = data.mean(axis=1)
    avg = avg.to_frame()
    avg = avg.T
    avg = avg.iloc[0][0]
    data = data - avg

    # Multiply the data to make it look better on a graph
    data = data * 10000

    # Apply median filter to reduce occational spikes in the data
    columns = len(data.columns)
    data = pd.DataFrame.to_numpy(data)
    data = np.reshape(data, columns)
    data = signal.medfilt(data, kernel_size=3)
    data = pd.DataFrame(data).T
    
    return data


def import_and_preprocess(files, label, pbar):
    li = []

    for file in files:
        data = pd.read_csv(file, index_col=False, header=None)
        data = preprocessing(data)
        li.append(data)
        pbar.update()

    data = pd.concat(li, axis=0, ignore_index=True, sort=False)
    data.insert(0, 'label', label)
    return data


def main():

    decimate_factor = 50

    data_folder = "../../data/william/"

    labels = "standing", "sitting"

    files = {key: glob.glob(f"{data_folder}/{key.capitalize()}//*.csv")
             for key in labels}

    n_files = [len(files[k]) for k in labels]
    for k, n in zip(labels, n_files):
        print(f"{k.capitalize()} N =", n)

    print("Importing and preprocessing data")

    with tqdm(total=sum(n_files)) as pbar:

        data_list = []

        for key, value in files.items():

            data_list.append(
                import_and_preprocess(pbar=pbar, files=value, label=key))

    print("Creating dataframe")
    df = pd.concat(data_list, axis=0, ignore_index=True, sort=False)
    print("N =", len(df))

    print("Normalizing data")

    x = df.iloc[:, 1:].values
    x = (x - x.min()) / (x.max() - x.min())
    x -= 0.5
    x *= 2
    # x -= x.mean()
    # x /= x.std()

    x = signal.decimate(x, decimate_factor, axis=1)

    df2 = pd.DataFrame(x)
    df2.insert(0, 'label', df.label)

    print("Writing file")
    f_name = f'{data_folder}/preprocessed_data.csv'
    df2.to_csv(f_name, index=True, header=True)


if __name__ == "__main__":
    main()

