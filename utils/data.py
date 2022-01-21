import os
import numpy as np
import librosa

from torch.utils.data import Dataset
from torchvision.transforms import Compose

from . import transforms


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
        window_len=63, conv=False,
        seed=1, rng=None,
        verbose=True):
        
        # Initialize random number generator
        if (rng is None):
            rng = np.random.default_rng(seed)
        
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
        
        sample = {'waveforms': {}}
        for key, data in self.waveforms.items():
            sample['waveforms'][key] = data[idx]
        
        sample['noised'] = self.transform(
            self.spectrograms['noised']['mags'][idx])
        sample['clean'] = self.transform(
            self.spectrograms['clean']['mags'][idx])
        sample['phase'] = self.spectrograms['noised']['phases'][idx]
        
        if (self.conv):
            for key, signal in sample.items():
                sample[key] = signal.expand((-1, 1, 1))
        
        return sample