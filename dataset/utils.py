import os
import json

import numpy as np
import librosa
from tqdm import tqdm

import torch
import torchaudio
from torch.utils.data import Dataset
from torchvision import transforms

from .audio import wav_to_mag_phase, mag_phase_to_wav

__all__ = [
    'get_data_split_idxs',
    'load_babble_dataset',
    'load_librispeech_subset',
    'get_random_slice',
    'add_noise_source',
    'create_synthetic_babble'
]


def get_data_split_idxs(N, test_size=.10, seed=1):
    # Create and shuffle indexes
    idxs = [i for i in range(N)]
    np.random.default_rng(seed).shuffle(idxs)
    # Set split size
    s = int(N*test_size + 1)
    # Create train and test indexes sets
    test_idxs = idxs[:s]
    train_idxs = idxs[s:]

    return train_idxs, test_idxs


def load_librispeech_subset(subset, data_root='data', data_dir='LibriSpeech'):
    return torchaudio.datasets.LIBRISPEECH(
        root=data_root,
        url=subset,
        folder_in_archive=data_dir,
        download=True)


def get_random_slice(waveform, slice_size, seed=0):
    max_idx = waveform.shape[0] - slice_size

    if (max_idx <= 0):
        return waveform

    rng = np.random.default_rng(seed)
    rint = rng.integers(low=0, high=max_idx, size=1)[0]
    return waveform[rint:(rint+slice_size)]


def add_noise_source(signal, noise, g=1, return_noise=True, seed=0):
    noise_slice = get_random_slice(noise, signal.shape[0], seed)
    noise = g*noise_slice
    noisy_signal = signal + noise

    if (return_noise):
        return noisy_signal, noise

    return noisy_signal


def create_synthetic_babble(ds, max_len=None, n_samples=30, flip_prob=0.3, seed=0, srate=16000):
    rng = np.random.default_rng(seed)
    rints = rng.integers(low=0, high=len(ds), size=n_samples)

    synth_babble = [ds[i][0].numpy()[0] for i in rints]

    flip_mask = rng.random(size=len(synth_babble)) + (flip_prob - 0.5)
    flip_mask = (flip_mask > 0.5).astype('int')
    for i, flip in enumerate(flip_mask):
        if (flip):
            synth_babble[i] = np.flip(synth_babble[i], axis=0)

    if (max_len is None):
        shapes = np.array([s.shape for s in synth_babble])
        max_len = shapes.max()

    samples = []
    for sample in synth_babble:
        sample = np.pad(
            sample[:max_len],
            (0, max_len - sample[:max_len].shape[0]))
        samples.append(sample)

    synth_babble = np.sum(np.vstack(samples), axis=0)
    return synth_babble


def create_dataset(
    libri_root, data_root,
    start_idx=0, end_idx=None,
    g_babble_range=None, g_room_range=None,
    n_samples_babble_range=None,
    n_iters=2, test=False,
    seed=1, srate=16000):


    libri_fmt = (
        'waveform', 'sample_rate',
        'utterance', 'speaker_id',
        'chapter_id', 'utterance_id')

    # Load LibriSpeech dataset
    libri_root = libri_root.split('/')
    libri = torchaudio.datasets.LIBRISPEECH(
        root='/'.join(libri_root[:-2]),
        url=libri_root[-1],
        folder_in_archive=libri_root[-2],
        download=True)

    # Load Room Noise Data
    noise_dir = 'data/RIRS_NOISES/real_rirs_isotropic_noises'
    fname = 'RVB2014_type1_noise_largeroom1_10.wav'
    room, sr = librosa.load(os.path.join(noise_dir, fname))

    # Create destination dataset directory
    os.makedirs(data_root, exist_ok=True)

    # Initialize random number generator
    rng = np.random.default_rng(seed=seed)

    # Set default dataset options
    if (g_babble_range is None):
        g_babble_range = np.linspace(0.05, 0.2, 4)
    if (g_room_range is None):
        g_room_range = np.linspace(0.5, 2, 4)
    if (n_samples_babble_range is None):
        n_samples_babble_range = np.arange(40, 71, 10)

    if (end_idx is None):
        end_idx = len(libri)

    # Initialize progress bar
    print('Creating dataset...')
    pbar = tqdm(
        total=(end_idx - start_idx),
        ascii=True, ncols=79,
        desc='Progress')

    # Main Loop
    for idx in range(start_idx, end_idx):

        # Load sample
        sample = libri[idx][0].numpy()[0]

        # Split samples
        n_splits = sample.shape[0]//srate
        rem = sample.shape[0]%srate

        if (rem == 0):
            sample_splits = np.array_split(sample, n_splits)
        else:
            sample_splits = np.array_split(sample[:-rem], n_splits)
            sample_splits += [np.pad(sample[-rem:], (0, srate - rem))]

        # Compute sample STFTs
        sample_mags, sample_phases = [], []
        for sample_split in sample_splits:
            sample_mag, sample_phase = wav_to_mag_phase(
                sample_split, n_fft=512, hop_len=64, win_len=512)
            sample_mag = np.pad(sample_mag, ((0, 0), (0, 2)))
            sample_phase = np.pad(sample_phase, ((0, 0), (0, 2)))
            sample_mags.append(sample_mag)
            sample_phases.append(sample_phase)

        sample_mags = np.asarray(sample_mags)
        sample_phases = np.asarray(sample_phases)

        for niter in range(n_iters):
            # Loop over all babble gains
            for g_babble in g_babble_range:
                # Create synthetic babble and add to sample
                n_babble_samples = rng.choice(n_samples_babble_range)
                synth_babble = create_synthetic_babble(
                    libri, max_len=sample.shape[0],
                    n_samples=n_babble_samples, flip_prob=0.4, srate=srate)

                noisy_sample, noise_b = add_noise_source(sample, synth_babble, g=g_babble)
                # Randomly choose a gain for room noise and add to noised sample
                g_room = rng.choice(g_room_range)
                noisy_sample, noise_r = add_noise_source(noisy_sample, room, g=g_room)
                noise = noise_b*g_babble + noise_r*g_room
                meta = {
                    'libri_sample_idx': idx,
                    'n_babble_samples': int(n_babble_samples),
                    'babble_gain': float(g_babble),
                    'room_gain': float(g_room),
                    'flip_prob': 0.4,
                    'srate': int(srate),
                    'filename': 'noised_s{}_n{}_gb{}_gr{}_{}'.format(
                        idx, n_babble_samples, g_babble, g_room, niter)
                }

                # Split noises
                if (rem == 0):
                    noise_splits = np.array_split(noise, n_splits)
                    noised_splits = np.array_split(noisy_sample, n_splits)
                else:
                    noise_splits = np.array_split(noise[:-rem], n_splits)
                    noise_splits += [np.pad(noise[-rem:], (0, srate - rem))]
                    noised_splits = np.array_split(
                        noisy_sample[:-rem], n_splits)
                    noised_splits += [np.pad(
                        noisy_sample[-rem:], (0, srate - rem))]

                # Compute noise stfts
                noise_mags, noise_phases = [], []
                for noise_split in noise_splits:
                    noise_mag, noise_phase = wav_to_mag_phase(
                        noise_split, n_fft=1024)
                    noise_mags.append(noise_mag)
                    noise_phases.append(noise_phase)

                noise_mags = np.asarray(noise_mags)
                noise_phases = np.asarray(noise_phases)

                noised_mags, noised_phases = [], []
                for noised_split in noised_splits:
                    noised_mag, noised_phase = wav_to_mag_phase(
                        noised_split, n_fft=512, hop_len=64, win_len=512)
                    noised_mags.append(noised_mag)
                    noised_phases.append(noised_phase)

                noised_mags = np.asarray(noised_mags)
                noised_phases = np.asarray(noised_phases)

                # Save data
                dirpath = os.path.join(data_root, meta['filename'])
                os.makedirs(dirpath, exist_ok=True)
                np.save(os.path.join(dirpath, 'noise.npy'), noise)
                np.save(os.path.join(dirpath, 'noised.npy'), noisy_sample)
                with open(os.path.join(dirpath, 'meta.json'), 'w') as fobj:
                    json.dump(meta, fobj)

                # Save original magnitudes
                np.save(
                    os.path.join(dirpath, 'clean_stft_mags.npy'),
                    sample_mags)
                # Save noised magnitudes
                np.save(
                    os.path.join(dirpath, 'noised_stft_mags.npy'),
                    noised_mags)
                # Save noised phases and noise magnitudes for test set
                if (test):
                    np.save(
                        os.path.join(dirpath, 'noised_stft_phases.npy'),
                        noised_phases)
                    np.save(
                        os.path.join(dirpath, 'noise_stft_mags.npy'),
                        noise_mags)

        pbar.update(1)
