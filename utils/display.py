import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Audio, display


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