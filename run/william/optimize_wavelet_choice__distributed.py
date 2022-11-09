import optuna
import pywt
import numpy as np

from main__wavelet_cnn import train


def objective(trial):
    wavelet_list = pywt.wavelist(kind='continuous')
    wavelet = trial.suggest_categorical('wavelet', wavelet_list)

    data_file = "../../data/william/dataset2/preprocessed_data__no_decimate.csv"

    metrics = train(
        wavelet=wavelet,
        data_file=data_file,
        fig_folder=None,
        seed=np.random.randint(2 ** 32 - 1),
        n_epochs=500,
        learning_rate=0.005,
        scales=np.geomspace(10, 520, num=20, dtype=int),
        dt=1,
        decimate=5,
        select_every=5,
        verbose=False)

    return metrics["accuracy_validation"]

#
# study2 = optuna.create_study(direction='maximize')
# study2.optimize(objective, n_trials=500)
#
# study2.best_params


if __name__ == '__main__':
    study = optuna.load_study(
        study_name='optimize_wavelet_choice', storage='mysql://root@localhost/optimize_wavelet_choice')
    study.optimize(objective, n_trials=500)