import numpy as np
from IPython.display import Audio, display
import matplotlib.pyplot as plt

import librosa
import soundfile as sf
import librosa.display


def save_audio(filepath, waveform, sample_rate):
    if (not isinstance(waveform, np.ndarray)):
        waveform = waveform.numpy()

    if (len(waveform.shape) == 1):
        waveform = np.expand_dims(waveform, axis=1)

    sf.write(filepath, waveform, sample_rate)


def play_audio(waveform, sample_rate):
    if (not isinstance(waveform, np.ndarray)):
        waveform = waveform.numpy()

    if (len(waveform.shape) == 1):
        waveform = np.expand_dims(waveform, axis=0)

    num_channels, num_frames = waveform.shape
    if (num_channels == 1):
        display(Audio(waveform[0], rate=sample_rate))
    elif (num_channels == 2):
        display(Audio((waveform[0], waveform[1]), rate=sample_rate))
    else:
        raise ValueError("Waveform with more than 2 channels are not supported.")


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


def show_split_sizes(loaders, names=('training', 'validation', 'testing')):
    for name, loader in zip(names, loaders):
        ns = len(loader)
        if (loader.test):
            ns = 0
            for batch in loader:
                ns += batch['magnitude'].shape[0]

        print('{:s} data: {:d} samples // {:.2f} mins'.format(
            name.title(), ns, ns/60))


def show_results(clean_wav, noised_wav, denoised_wav, srate=16000):

    print('Clean Signal')
    play_audio(clean_wav, srate)
    print('Noised Signal')
    play_audio(noised_wav, srate)
    print('Denoised Signal')
    play_audio(denoised_wav, srate)

    # Display Results
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(20, 15))

    librosa.display.waveshow(clean_wav, sr=srate, color='b', ax=axes[0][0])
    axes[0][0].set(title='Original Waveform', xlabel=None)

    librosa.display.waveshow(noised_wav, sr=srate, color='r', ax=axes[1][0])
    axes[1][0].set_title('Noised Waveform')

    librosa.display.waveshow(denoised_wav, sr=srate, color='r', ax=axes[2][0])
    axes[2][0].set_title('Denoised Reconstructed Waveform')


    D = librosa.amplitude_to_db(
        np.abs(librosa.stft(clean_wav)), ref=np.max)
    im = librosa.display.specshow(
        D, y_axis='linear', sr=srate,
        hop_length=512, x_axis='time', ax=axes[0][1])
    axes[0][1].set_title('Original Clean Spectrogram')
    fig.colorbar(im, ax=axes[0][1], format="%+2.f dB")

    D = librosa.amplitude_to_db(
        np.abs(librosa.stft(noised_wav)), ref=np.max)
    im = librosa.display.specshow(
        D, y_axis='linear', sr=srate,
        hop_length=512, x_axis='time', ax=axes[1][1])
    axes[1][1].set_title('Noised Spectrogram')
    fig.colorbar(im, ax=axes[1][1], format="%+2.f dB")

    D = librosa.amplitude_to_db(
        np.abs(librosa.stft(denoised_wav)), ref=np.max)
    im = librosa.display.specshow(
        D, y_axis='linear', sr=srate,
        hop_length=512, x_axis='time', ax=axes[2][1])
    axes[2][1].set_title('Denoised Spectrogram')
    fig.colorbar(im, ax=axes[2][1], format="%+2.f dB")

    return fig, axes
