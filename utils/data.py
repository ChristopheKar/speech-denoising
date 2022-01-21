import os
import numpy as np
import librosa

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose

from . import transforms


def load_data(
    libri_path='data/LibriSpeech/dev-clean',
    babble_path='data/babble.wav',
    train_mins=60, val_mins=10, test_mins=10,
    srate=16000, batch_size=8,
    n_fft=512, hop_len=256, gains=[0.2],
    conv=False, pin_memory=False):

    # Set size variables
    train_idx = int(srate*60*train_mins)
    val_idx = int(srate*60*val_mins)
    test_idx = int(srate*60*test_mins)
    max_minutes = train_mins + val_mins + test_mins
    max_samples = int(srate*60*max_minutes)

    # Load speech data
    speech = load_libri(libri_path, srate, max_samples)

    train_data = BabbledLibri(
        speech[:train_idx],
        srate=srate,
        babble_path=babble_path,
        babble_gains=gains,
        n_fft=n_fft, hop_len=hop_len,
        conv=conv)

    train_dl = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory)

    val_data = BabbledLibri(
        speech[train_idx:(train_idx + val_idx)],
        srate=srate,
        babble_path=babble_path,
        babble_gains=gains,
        n_fft=n_fft, hop_len=hop_len,
        conv=conv)

    val_dl = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory)

    test_data = BabbledLibri(
        speech[(train_idx + val_idx):(train_idx + val_idx + test_idx)],
        srate=srate,
        babble_path=babble_path,
        babble_gains=gains,
        n_fft=n_fft, hop_len=hop_len,
        conv=conv)

    test_dl = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory)

    return train_data, train_dl, val_data, val_dl, test_data, test_dl


def load_babble(babble_path, srate=16000):
    babble, _ = librosa.load(babble_path, sr=srate)
    return babble


def load_libri(libri_path, srate=16000, max_samples=None):
    speech = []
    n_samples = 0
    for root, dirs, files in os.walk(libri_path):
        for filename in files:
            if (filename.endswith('.flac')):
                filepath = os.path.join(root, filename)
                data, _ = librosa.load(filepath, sr=srate)
                n_samples += data.shape[0]
                speech.append(data)
                if ((max_samples) and (n_samples > max_samples)):
                    return np.hstack(speech)[:max_samples]

    speech = np.hstack(speech)
    return speech


def generate_babbled_speech(speech, babble, gain=1.):
    n_repeats = 1 + speech.shape[0]//babble.shape[0]
    babble = np.hstack([babble]*n_repeats)[:speech.shape[0]]
    noise = gain*babble
    noised_speech = speech + noise
    return noised_speech, noise



class BabbledLibri(Dataset):

    def __init__(
        self,
        speech,
        srate=16000,
        babble_path='data/babble.wav',
        babble_gains=[0.2],
        c=1e-7,
        n_fft=512, hop_len=256,
        conv=False):

        self.srate = srate

        # Load babble data
        babble = load_babble(babble_path, srate)

        # Add babble noise to speech
        for gain in babble_gains:
            noised_speech, noise = generate_babbled_speech(
                speech, babble, gain=gain)

        # Split signals into 1-second windows
        n_splits = speech.shape[0]//srate
        self.waveforms = {}
        self.waveforms['clean'] = np.array_split(
            speech[:n_splits*srate], n_splits)
        self.waveforms['noised'] = np.array_split(
            noised_speech[:n_splits*srate], n_splits)
        self.waveforms['noise'] = np.array_split(
                    noise[:n_splits*srate], n_splits)

        # Convert signals to spectrogram
        self.spectrograms = {}
        for key, signals in self.waveforms.items():
            self.spectrograms[key] = {'mags': [], 'phases': []}
            for wav in signals:
                mag, phase = transforms.wav_to_mag_phase(
                    wav, n_fft=n_fft, hop_len=hop_len, win_len=n_fft)
                self.spectrograms[key]['mags'].append(mag)
                self.spectrograms[key]['phases'].append(phase)

        # Initialize transforms
        self.tensor = transforms.ToTensor()
        self.log = transforms.LogTransformer(c=1e-7)
        self.resizer = transforms.Resizer()

        # Set min and max scaling values
        max_vals, min_vals = [], []
        for mag in self.spectrograms['noised']['mags']:
            max_vals.append(self.log(mag).max(axis=1))
            min_vals.append(self.log(mag).min(axis=1))

        self.max_vals = np.max(max_vals, axis=0)
        self.min_vals = np.min(min_vals, axis=0)

        # Initialize scaler
        self.scaler = transforms.MinMaxScaler(self.min_vals, self.max_vals)

        # Create forward and inverse transforms
        self.transform = Compose([
            self.log, self.scaler, self.resizer, self.tensor])
        self.inv_transform = lambda x: self.log.revert(
            self.scaler.revert(self.resizer.revert(self.tensor.revert(x))))

        self.conv = conv
        self.n_samples = n_splits


    def __len__(self):
        """Return length of data set."""
        return self.n_samples


    def __getitem__(self, idx):

        sample = {}
        sample['noised'] = self.transform(
            self.spectrograms['noised']['mags'][idx])
        sample['clean'] = self.transform(
            self.spectrograms['clean']['mags'][idx])
        sample['phase'] = self.spectrograms['noised']['phases'][idx]

        if (self.conv):
            sample['clean'] = sample['clean'].expand((1, -1, -1))
            sample['noised'] = sample['noised'].expand((1, -1, -1))

        # Add waveforms to sample
        sample['waveforms'] = {}
        for key, data in self.waveforms.items():
            sample['waveforms'][key] = data[idx]

        return sample
