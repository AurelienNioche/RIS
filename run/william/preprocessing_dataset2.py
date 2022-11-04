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
    data -= avg

    # Multiply the data to make it look better on a graph
    data = data * 10000

    # Apply median filter to reduce occational spikes in the data
    columns = len(data.columns)
    data = data.to_numpy()
    data = np.reshape(data, columns)
    data = signal.medfilt(data, kernel_size=3)
    data = pd.DataFrame(data).T
    
    return data


def import_and_preprocess(files, label, position, pbar):
    li = []

    for file in files:
        data = pd.read_csv(file, index_col=False, header=None)
        data = preprocessing(data)
        li.append(data)
        pbar.update()

    data = pd.concat(li, axis=0, ignore_index=True, sort=False)
    data.insert(0, 'label', label)
    data.insert(0, 'position', position)
    return data


def main():

    decimate_factor = 50

    data_folder = "../../data/william/dataset2"

    labels = "standing", "sitting"

    positions = ["position1", "position2", "position3"]

    files = {pos: {lab: glob.glob(f"{data_folder}/{pos.capitalize()}/{lab.capitalize()}/*.csv")
             for lab in labels} for pos in positions}

    n_files = 0
    for pos in positions:
        for lab in labels:
            n = len(files[pos][lab])
            n_files += n
            print(f"{pos}-{lab} N = {n}")

    print(f"Total N = {n_files}")

    print("Importing and preprocessing data")

    with tqdm(total=n_files) as pbar:

        data_list = []

        for pos, value in files.items():
            for lab, file_list in value.items():

                data_list.append(
                    import_and_preprocess(pbar=pbar, files=file_list, label=lab, position=pos))

    print("Creating dataframe")
    df = pd.concat(data_list, axis=0, ignore_index=True, sort=False)
    print("N =", len(df))

    print("Normalizing data")

    len_data = len(df)
    data = df.iloc[:, 2:].values
    for i in range(len_data):
        x = data[i]
        x = (x - x.min()) / (x.max() - x.min())
        x -= 0.5
        x *= 2
        data[i] = x

    if decimate_factor is None:
        add = "__no_decimate"
    else:
        add = ""
        data = signal.decimate(data, decimate_factor, axis=1)

        df_temp = pd.DataFrame(data)
        df_temp.insert(0, 'label', df.label)
        df_temp.insert(0, 'position', df.position)

        df = df_temp

    print("Writing file")
    f_name = f'{data_folder}/preprocessed_data{add}.csv'
    df.to_csv(f_name, index=True, header=True)


if __name__ == "__main__":
    main()

