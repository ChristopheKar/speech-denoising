import os

import numpy as np
import librosa
import torch
import torchaudio
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from .audio import wav_to_mag_phase, mag_phase_to_wav
from . import transforms


class NoisyLibriSpeechDataset(Dataset):
    """Artificially noised LibriSpeech dataset."""

    def __init__(
        self,
        data_root, libri_root,
        include_idxs=[], exclude_idxs=[], size=None,
        test=False, conv=False, seed=1, srate=16000):

        self.srate = srate

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
        self.resizer = transforms.ResizeMagnitude(
            h=self.target_h, w=self.target_w)
        self.unsizer = transforms.ResizeMagnitude(
            h=self.init_h, w=self.init_w)
        self.scaler = transforms.MinMaxScaler()

        self.transform = Compose([
            self.resizer,
            self.scaler,
            transforms.ToTensor()])

        self.inv_transform = Compose([
            transforms.FromTensor(),
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
                item['phase'] = transforms.FromTensor()(self.squeeze(
                    item['phase'], force=self.test))
            else:
                item['phase'] = np.squeeze(item['phase'])
        else:
            return self.inv_transform(self.squeeze(item))

        return item
