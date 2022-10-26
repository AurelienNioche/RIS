import torch
import glob
import os
from adversarial_autoencoder import Decoder, Encoder

def main():

    root_folder = "../.."
    bkp_folder = f"{root_folder}/bkp/william/generative_models"
    conditions = [x[0].split("/")[-1] for x in os.walk(bkp_folder)][1:]
    for cond in conditions:
        folder = f"{bkp_folder}/{cond}"
        encoder = Encoder.load(folder=folder)
        decoder = Decoder.load(folder=folder)
        n = 10000
        z = torch.randn((n, encoder.latent_dim))
        samples = decoder(z)


if __name__ == "__main__":
    main()
