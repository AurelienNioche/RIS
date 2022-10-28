import torch
import pandas as pd
import os
from adversarial_autoencoder import Decoder, Encoder


def main():

    file_for_export = f'../../data/william/generated_data.csv'
    bkp_folder = f"../../bkp/william/generative_models"
    conditions = [x[0].split("/")[-1] for x in os.walk(bkp_folder)][1:]

    n = 10000

    df_list = []

    for cond in conditions:
        folder = f"{bkp_folder}/{cond}"
        encoder = Encoder.load(folder=folder)
        decoder = Decoder.load(folder=folder)
        z = torch.randn((n // 2, encoder.latent_dim))
        samples = decoder(z).detach().numpy()

        df = pd.DataFrame(samples)
        df.insert(0, 'label', cond)

        df_list.append(df)

    gen_data = pd.concat(df_list, axis=0, ignore_index=True, sort=False)
    gen_data.to_csv(file_for_export, index=True, header=True)


if __name__ == "__main__":
    main()
