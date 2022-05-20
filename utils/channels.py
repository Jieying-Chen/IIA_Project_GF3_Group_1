import numpy as np
from .audio import wav_to_signal
import scipy.signal

def channel_output(h, x, noise=False, sigma=0.1):
    """Return the convolution of h and x, with white gaussian noise added

    :param h: impulse response of channel in time domain
    :param x: input signal in time domain
    :param noise: add noise if True, defaults to False
    :param sigma: standard deviation of gaussian noise, defaults to 0.1
    :return: output signal
    """

    y = np.convolve(h, x)
    if noise:
        add_noise = sigma * np.random.rand(y.size)
        return y + add_noise
    else:
        return y

def octagon(path, fs):
    """Read path/octagon.wav, resample to fs Hz and return the resampled signal"""

    data, samplerate = wav_to_signal("octagon", path)
    secs = data.size / samplerate
    samps = int(secs * fs)

    return scipy.signal.resample(data, samps)