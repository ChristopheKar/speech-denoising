import numpy as np
from dataset.utils import create_dataset


if __name__ == '__main__':
    libri_root = 'data/LibriSpeech/dev-clean'
    seed = 1
    srate = 16000

    data_dir = 'data/noised_synth_babble'

    # Dataset Options
    g_babble_range = np.linspace(0.05, 0.2, 5)
    g_room_range = np.linspace(0.5, 2, 4)
    n_samples_babble_range = np.arange(10, 71, 10)
    n_iters = 2
    N = 20
    save_noise = False
    save_phase = False

    create_dataset(
        libri_root, data_dir, N=N,
        g_babble_range=g_babble_range,
        g_room_range=g_room_range,
        n_samples_babble_range=n_samples_babble_range,
        n_iters=n_iters,
        save_noise=save_noise,
        save_phase=save_phase,
        seed=seed,
        srate=srate)
