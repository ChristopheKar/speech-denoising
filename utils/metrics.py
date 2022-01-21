import numpy as np


def power(sig):
    """Compute signal power as sum(s^2)/N"""
    return np.sum(np.square(sig))/sig.size


def snr(sig, noise):              
    """Signal to Noise Ratio: Pure Signal Power / Pure Noise Power"""
    return 10*np.log10(power(sig)/power(noise))


def sdr(sig, sig_pred):
    """
    Signal Distortion Ratio: Pure Signal Power / Residual Noise Power
    with Residual Noise Power = Power(Predicted_Signal - Pure_Signal)
    """
    return 10*np.log10(power(sig)/power(sig_pred - sig))


def gain(sig, noise, sig_est):
    """Compute gain from reference signal, noise, and estimated signal"""
    return sdr(sig, sig_est) - snr(sig, noise)


def rmse(sig, sig_est):
    """Compute reconstruction normalized root-mean-squared error."""
    div = np.square(sig)
    div[div < 1e-10] = 1e-10
    return np.sqrt(np.square(sig - sig_est)/div)