import numpy as np
import librosa


def wav_to_mag_phase(waveform, n_fft=1024, hop_len=64, win_len=512, window='hamming'):
    """Convert waveform to Magnitude and Phase through STFT"""

    spectrogram = librosa.stft(
        waveform, n_fft=n_fft, hop_length=hop_len, 
        win_length=win_len, window=window)
    
    mag, phase = librosa.magphase(spectrogram)
    mag_db = librosa.amplitude_to_db(mag, ref=np.max)

    return mag, phase


def mag_phase_to_wav(mag, phase=None, hop_len=64, length=None, win_len=512, window='hamming'):
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