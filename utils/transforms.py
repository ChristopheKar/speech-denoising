import numpy as np
import torch
import librosa


def wav_to_mag_phase(
    waveform, n_fft=512, hop_len=256, 
    win_len=512, window='hamming'):
    
    """Convert waveform to Magnitude and Phase through STFT"""

    spectrogram = librosa.stft(
        waveform, n_fft=n_fft, hop_length=hop_len,
        win_length=win_len, window=window)

    mag = np.abs(spectrogram)
    phase = np.angle(spectrogram)

    return mag, phase


def mag_phase_to_wav(
    mag, phase=None,
    hop_len=256, length=None,
    win_len=512, window='hamming'):
    
    """Convert Magnitude and Phase to Waveform through Inverse-STFT"""
    if (phase is None):
        waveform = librosa.griffinlim(
            mag, hop_length=hop_len, length=length,
            win_length=win_len, window=window)
    else:
        spectrogram = mag*np.exp(1j*phase)
        waveform = librosa.istft(
            spectrogram, hop_length=hop_len, length=length,
            win_length=win_len, window=window)

    return waveform


class LogTransformer:
    """Transform data using logarithm, adding a small constant."""
    
    def __init__(self, c=1e-7):
        self.c = c
        
    
    def __call__(self, x):
        return np.log(x + self.c)
    
    
    def revert(self, x):
        return (np.exp(x) - self.c)


class MinMaxScaler:
    """Normalize array to [0, 1] range by min-max scaling."""
    
    def __init__(self, min_vals, max_vals):
        super().__init__()
        self.max = max_vals
        self.min = min_vals
        self.diff = max_vals - min_vals
        if (isinstance(self.diff, np.ndarray)):
            self.diff[self.diff == 0] = 1.


    def __call__(self, x):
        return ((x.T - self.min)/self.diff).T
    
    
    def revert(self, x):
        return ((x.T*self.diff) + self.min).T


class ToTensor:
    """Convert numpy.ndarrays to torch.Tensor."""
    
    def __init__(self):
        pass

    def __call__(self, x):
        return torch.from_numpy(x)
    
    
    def revert(self, x):
        return x.detach().cpu().numpy()
    
    
class Resizer:
    """Resize spectrogram magnitude by truncating
    from row-dimension and zero-padding to col-dimension."""
    
    def __init__(self):
        pass
    
    
    def __call__(self, x):
        return np.pad(x[:-1,:], [(0, 0), (0, 1)])
    
    
    def revert(self, x):
        return np.pad(x, [(0, 1), (0, 0)])[:, :-1]