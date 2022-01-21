import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from IPython.display import Audio, display


def save_audio(filepath, waveform, srate=16000):
    if (not isinstance(waveform, np.ndarray)):
        waveform = waveform.numpy()

    if (len(waveform.shape) == 1):
        waveform = np.expand_dims(waveform, axis=1)

    print(waveform.shape)

    sf.write(filepath, waveform, srate)


def play_audio(waveform, srate=16000):
    display(Audio(np.squeeze(waveform), rate=srate))


def plot_losses(ax, hist, label='Loss', title=None):
    ax.plot(hist['losses'], label='Training Loss')
    ax.plot(hist['val_losses'], label='Validation Loss')
    ax.set_ylabel(label)
    ax.set_xlabel('Epochs')
    ax.legend()
    if (title is None):
        title = 'Loss Function'
    ax.set_title(title)
    return ax
