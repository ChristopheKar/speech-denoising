import os
import json

import numpy as np
import librosa

import torch
import torchaudio
from torch.utils.data import Dataset
from torchvision import transforms

from audio import wav_to_mag_phase, mag_phase_to_wav


def get_data_split_idxs(N, test_size=.10, seed=1):
    idxs = [i for i in range(N)]
    np.random.default_rng(seed).shuffle(idxs)

    s = int(N*test_size + 1)

    val_idxs = idxs[:s]
    test_idxs = idxs[s:2*s]
    train_idxs = idxs[2*s:]

    return train_idxs, val_idxs, test_idxs


class NoisyLibriSpeechDataset(Dataset):
    """Artificially noised LibriSpeech dataset."""

    def __init__(
        self,
        data_root, libri_root,
        include_idxs=[], exclude_idxs=[], size=None,
        test=False, conv=False, seed=1):

        # Initialize random number generator
        self.rng = np.random.default_rng(seed)

        # Load corresponding LibriSpeech dataset
        libri_root = libri_root.split('/')
        self.libri = torchaudio.datasets.LIBRISPEECH(
            root='/'.join(libri_root[:-2]),
            url=libri_root[-1],
            folder_in_archive=libri_root[-2],
            download=True)

        # Load available LibriSpeech indices
        self.data_root = data_root
        dirnames = np.asarray(os.listdir(data_root))
        libri_idxs = [self._name_to_libri_idx(name) for name in dirnames]

        # If there are no provided indices to include,
        # randomly sample `size` indices
        if (len(include_idxs) == 0):

            # Set size value if not specified
            if ((size is None) or (size > libri_idxs.size)):
                self.size = libri_idxs.size
            else:
                self.size = size

            # Remove excluded indices from possible choices
            libri_idxs = [i for i in libri_idxs if (i not in exclude_idxs)]

            # Randomly select subset of possible indexes
            self.subset_idxs = self.rng.choice(
                np.unique(libri_idxs),
                size=self.size, replace=False, shuffle=False)

        # Use provided indices in dataset
        else:
            self.subset_idxs = np.asarray(include_idxs)

        # Choose data directories corresponding to chosen indices subset
        self.directories = []
        self.libri_idxs = []
        self.sizes = []
        for idx in self.subset_idxs:
            for dirname in dirnames:
                libri_idx = self._name_to_libri_idx(dirname)
                if (idx == libri_idx):
                    self.directories.append(dirname)
                    self.libri_idxs.append(libri_idx)
                    self.sizes.append(self._get_n_samples(
                        len(self.directories) - 1))

        # Shuffle data
        random_idxs = np.arange(len(self.directories))
        self.rng.shuffle(random_idxs)
        self.directories = list(np.asarray(self.directories)[random_idxs])
        self.libri_idxs = list(np.asarray(self.libri_idxs)[random_idxs])
        self.sizes = list(np.asarray(self.sizes)[random_idxs])
        self.cumsizes = np.cumsum(self.sizes)

        # Set actual dataset size
        self.test = test
        if (self.test):
            self.size = len(self.directories)
        else:
            self.size = np.sum(self.sizes)


        # If conv is True, expand data dimensions to 1x256x256
        self.conv = conv

        # Initialize data shapes
        self.target_h, self.target_w = (256, 256)
        self.target_shape = (self.target_h, self.target_w)
        self.init_h, self.init_w = self.__getitem__(
            0, transform=False)['magnitude'].shape[-2:]

        # Initialize transforms
        self.resizer = ResizeMagnitude(h=self.target_h, w=self.target_w)
        self.unsizer = ResizeMagnitude(h=self.init_h, w=self.init_w)
        self.scaler = MinMaxScaler()

        self.transform = transforms.Compose([
            self.resizer,
            self.scaler,
            ToTensor()])

        self.inv_transform = transforms.Compose([
            FromTensor(),
            self.scaler.unscale,
            self.unsizer])


    def _get_n_samples(self, idx):
        return np.load(os.path.join(
            self.data_root,
            self.directories[idx],
            'sample_stft_mags.npy')).shape[0]


    def _name_to_libri_idx(self, name):
        return int(name.split('_')[1][1:])


    def __len__(self):
        """Returns length of data set."""
        return self.size


    def __getitem__(self, idx, transform=True):

        if (self.test):
            dir_idx = idx
            sample_idx = slice(None)
        else:
            dir_idx = np.argwhere(self.cumsizes > idx)[0][0]
            if (dir_idx == 0):
                sample_idx = idx
            else:
                sample_idx = idx - self.cumsizes[dir_idx - 1]
        libri_idx = self.libri_idxs[dir_idx]
        dirname = self.directories[dir_idx]
        dirpath = os.path.join(self.data_root, dirname)

        clean_mag = np.load(os.path.join(dirpath, 'sample_stft_mags.npy'))
        noised_mag = np.load(os.path.join(dirpath, 'noised_stft_mags.npy'))
        noised_phase = np.load(os.path.join(dirpath, 'noised_stft_phases.npy'))

        if (transform):
            noised_mag = self.transform(noised_mag)
            clean_mag = self.transform(clean_mag)

        noised_mag = noised_mag[sample_idx, :, :]
        noised_phase = noised_phase[sample_idx, :, :]
        clean_mag = clean_mag[sample_idx, :, :]

        # Reshape to fit use-case:
        # pixels: 256x256
        # conv: 1x256x256
        if (self.conv):
            if (self.test):
                ax = 1
                shape = (1, -1, -1, -1)
            else:
                ax = 0
                shape = (1, -1, -1)

            noised_phase = np.expand_dims(noised_phase, axis=ax)
            if (transform):
                noised_mag = noised_mag.expand(shape)
                clean_mag = clean_mag.expand(shape)
                if (self.test):
                    noised_mag = noised_mag.transpose(1, 0)
                    clean_mag = clean_mag.transpose(1, 0)
            else:
                noised_mag = np.expand_dims(noised_mag, axis=ax)
                clean_mag = np.expand_dims(clean_mag, axis=ax)

        item = {
            'libri_index': libri_idx,
            'magnitude': noised_mag,
            'phase': noised_phase,
            'target': clean_mag
        }

        return item


    def squeeze(self, x, force=False):
        if (((len(x.shape) == 4) and (not self.conv)) or (force)):
            return x.squeeze()
        else:
            return x


    def spec_to_wav(self, magnitudes, phases):
        wav = []
        for i in range(magnitudes.shape[0]):
            wav.extend(mag_phase_to_wav(
                magnitudes[i, :, :], phases[i, :, :],
                hop_len=64, win_len=512))
        return np.asarray(wav)


    def restore(self, item):
        if (isinstance(item, dict)):
            if (isinstance(item['libri_index'], torch.Tensor)):
                item['libri_index'] = int(item['libri_index'].detach())

            item['magnitude'] = self.inv_transform(self.squeeze(
                item['magnitude'], force=self.test))
            item['target'] = self.inv_transform(self.squeeze(
                item['target'], force=self.test))
            if (not isinstance(item['phase'], np.ndarray)):
                item['phase'] = FromTensor()(self.squeeze(
                    item['phase'], force=self.test))
            else:
                item['phase'] = np.squeeze(item['phase'])
        else:
            return self.inv_transform(self.squeeze(item))

        return item


class FromTensor(object):
    """Convert torch Tensor to numpy ndarray."""

    def __call__(self, x):
        return x.detach().numpy()


class ToTensor(object):
    """Convert numpy.ndarrays dict containing tensor and metadata."""

    def __call__(self, x):
        return torch.from_numpy(x)


class ResizeMagnitude:
    """Resize STFT Magnitude by cutting or padding to square-size."""

    def __init__(self, h=None, w=None):
        super().__init__()

        self.h = h
        self.w = w


    def __call__(self, x):

        if (self.w is None):
            # Zero-pad to nearest upper power of 2
            pad_width = self.nearest_2pow(x.shape[2]) - x.shape[2]
            x = np.pad(x, ((0, 0), (0, 0), (0, pad_width)))
        elif (self.w > x.shape[2]):
            # Zero-pad to original size
            pad_width = self.w - x.shape[2]
            x = np.pad(x, ((0, 0), (0, 0), (0, pad_width)))
        else:
            # Truncate to original size
            x = x[:, :, :self.w]

        if (self.h is None):
            # Zero-pad to nearest upper power of 2
            pad_width = self.nearest_2pow(x.shape[1]) - x.shape[1]
            x = np.pad(x, ((0, 0), (0, pad_width), (0, 0)))
        elif (self.h > x.shape[1]):
            # Zero-pad to original size
            pad_width = self.h - x.shape[1]
            x = np.pad(x, ((0, 0), (0, pad_width), (0, 0)))
        else:
            # Truncate to original size
            x = x[:, :self.h, :]

        return x


    def __repr__(self):
        return self.__class__.__name__ + "()"


    def nearest_2pow(self, x):
        return 2**np.ceil(np.log10(x)/np.log10(2)).astype('int')


class MinMaxScaler:
    """Normalize array to [0, 1] range by min-max scaling."""

    def __init__(self, max_val=None, min_val=None):
        super().__init__()
        self.max = max_val
        self.min = min_val


    def __call__(self, x):
        if (self.max is None):
            self.max = x.max()

        if (self.min is None):
            self.min = x.min()

        return (x - self.min)/(self.max - self.min)


    def __repr__(self):
        return self.__class__.__name__ + "()"


    def unscale(self, x):
        if (self.max is None):
            raise ValueError('Cannot unscale: max value is undefined')

        if (self.min is None):
            self.min = 0

        return x*(self.max - self.min) + self.min


def load_babble_dataset(indexes, data_root='data/noised_synth_babble'):
    data_dirs = {d.split('_')[1]: d for d in os.listdir(data_root)}
    data_dirs = [data_dirs['s{}'.format(idx)] for idx in indexes]
    sample_stft_mags = []
    noised_stft_mags = []
    noised_stft_phases = []
    for data_dir in data_dirs:
        sample_stft_mags.append(np.load(os.path.join(data_root, data_dir, 'sample_stft_mags.npy')))
        noised_stft_phases.append(np.load(os.path.join(data_root, data_dir, 'noised_stft_phases.npy')))
        noised_stft_mags.append(np.load(os.path.join(data_root, data_dir, 'noised_stft_mags.npy')))

    sample_stft_mags = np.vstack(sample_stft_mags)
    noised_stft_mags = np.vstack(noised_stft_mags)
    noised_stft_phases = np.vstack(noised_stft_phases)

    return sample_stft_mags, noised_stft_mags, noised_stft_phases


def load_librispeech_subset(subset_name, data_root='data', data_dir='LibriSpeech'):
    return torchaudio.datasets.LIBRISPEECH(
        root=data_root,
        url=subset_name,
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

    synth_babble = np.vstack([np.pad(s[:max_len], (0, max_len - s[:max_len].shape[0])) for s in synth_babble])
    return np.sum(synth_babble, axis=0)


if __name__ == '__main__':

    librispeech_fmt = (
        'waveform', 'sample_rate',
        'utterance', 'speaker_id',
        'chapter_id', 'utterance_id')

    data_root = 'data'
    data_dir = 'LibriSpeech'
    subset_name = 'dev-clean'
    seed = 1

    srate = 16000

    ds = load_librispeech_subset(subset_name, data_root, data_dir)
    N = len(ds)
    print('Dataset Size: {}'.format(N))

    # Load Room Noise Data
    data_dir = 'RIRS_NOISES/real_rirs_isotropic_noises'
    fname = 'RVB2014_type1_noise_largeroom1_10.wav'
    room, sr = librosa.load(os.path.join(data_root, data_dir, fname))

    # Dataset Options
    g_babble_range = np.linspace(0.05, 0.2, 5)
    g_room_range = np.linspace(0.5, 2, 4)
    n_samples_babble_range = np.arange(10, 71, 10)
    niters = 2

    data_dir = 'data/noised_synth_babble'
    os.makedirs(data_dir, exist_ok=True)

    rng = np.random.default_rng(seed=seed)


    N = len(ds)
    N = 20

    # Main Loop
    for idx in range(N):

        sample = ds[idx][0].numpy()[0]

        # split samples
        n_splits, rem = sample.shape[0]//srate, sample.shape[0]%srate
        if (rem == 0):
            sample_splits = np.array_split(sample, n_splits)
        else:
            sample_splits = np.array_split(sample[:-rem], n_splits) +  [np.pad(sample[-rem:], (0, srate - rem))]

        # compute sample stfts
        sample_mags, sample_phases = [], []
        for sample_split in sample_splits:
            sample_mag, sample_phase = wav_to_mag_phase(sample_split, n_fft=512, hop_len=64, win_len=512)
            sample_mag, sample_phase=  np.pad(sample_mag, ((0, 0), (0, 2))), np.pad(sample_phase, ((0, 0), (0, 2)))
            sample_mags.append(sample_mag)
            sample_phases.append(sample_phase)
        sample_mags, sample_phases = np.asarray(sample_mags), np.asarray(sample_phases)

        for niter in range(niters):
            # loop over all babble gains
            for g_babble in g_babble_range:
                # create synthetic babble and add to sample
                n_babble_samples = rng.choice(n_samples_babble_range)
                synth_babble = create_synthetic_babble(
                    ds, max_len=sample.shape[0],
                    n_samples=n_babble_samples, flip_prob=0.4, srate=srate)

                noisy_sample, noise_b = add_noise_source(sample, synth_babble, g=g_babble)
                # randomly choose a gain for room noise and add to noised sample
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
                    'filename': f'noised_s{idx}_n{n_babble_samples}_gb{g_babble}_gr{g_room}_{niter}'
                }

                # split noises
                if (rem == 0):
                    noise_splits = np.array_split(noise, n_splits)
                    noised_splits = np.array_split(noisy_sample, n_splits)
                else:
                    noise_splits = np.array_split(noise[:-rem], n_splits) +  [np.pad(noise[-rem:], (0, srate - rem))]
                    noised_splits = np.array_split(noisy_sample[:-rem], n_splits) +  [np.pad(noisy_sample[-rem:], (0, srate - rem))]

                # compute noise stfts
                noise_mags, noise_phases = [], []
                for noise_split in noise_splits:
                    noise_mag, noise_phase = wav_to_mag_phase(noise_split, n_fft=1024)
                    noise_mags.append(noise_mag)
                    noise_phases.append(noise_phase)
                noise_mags, noise_phases = np.asarray(noise_mags), np.asarray(noise_phases)

                noised_mags, noised_phases = [], []
                for noised_split in noised_splits:
                    noised_mag, noised_phase = wav_to_mag_phase(noised_split, n_fft=512, hop_len=64, win_len=512)
                    noised_mags.append(noised_mag)
                    noised_phases.append(noised_phase)
                noised_mags, noised_phases = np.asarray(noised_mags), np.asarray(noised_phases)


                # save data
                dirpath = os.path.join(data_dir, meta['filename'])
                os.makedirs(dirpath, exist_ok=True)
                np.save(os.path.join(dirpath, 'noise.npy'), noise)
                np.save(os.path.join(dirpath, 'noised.npy'), noisy_sample)
                with open(os.path.join(dirpath, 'meta.json'), 'w') as fobj:
                    json.dump(meta, fobj)

                np.save(os.path.join(dirpath, 'sample_stft_mags.npy'), sample_mags)
                np.save(os.path.join(dirpath, 'sample_stft_phases.npy'), sample_phases)

                np.save(os.path.join(dirpath, 'noise_stft_mags.npy'), noise_mags)
                np.save(os.path.join(dirpath, 'noise_stft_phases.npy'), noise_phases)

                np.save(os.path.join(dirpath, 'noised_stft_mags.npy'), noised_mags)
                np.save(os.path.join(dirpath, 'noised_stft_phases.npy'), noised_phases)

        print('Progress: {}/{}'.format(idx+1, N), end='\r')
