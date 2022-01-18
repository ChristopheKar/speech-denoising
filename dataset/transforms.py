import numpy as np
import torch
from .audio import wav_to_mag_phase, mag_phase_to_wav

__all__ = [
    'FromTensor',
    'ToTensor',
    'ResizeMagnitude',
    'MinMaxScaler'
]


class FromTensor(object):
    """Convert torch Tensor to numpy ndarray."""

    def __call__(self, x):
        return x.detach().cpu().numpy()


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

    def __init__(self, max_val, min_val):
        super().__init__()
        self.max = max_val
        self.min = min_val
        self.diff = max_val - min_val
        if (isinstance(self.diff, np.ndarray)):
            self.diff[self.diff == 0] = 1.


    def __call__(self, x):
        return (np.log1p(x) - self.min)/self.diff


    def __repr__(self):
        return self.__class__.__name__ + "()"


    def unscale(self, x):
        if (self.max is None):
            raise ValueError('Cannot unscale: max value is undefined')

        if (self.min is None):
            self.min = 0

        return np.expm1(x)*self.diff + self.min
