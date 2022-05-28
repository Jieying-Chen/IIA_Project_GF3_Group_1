import numpy as np
import scipy.signal

def generate_chirp(duration, fs, low=20, high=20000, silence_duration=0, double=False):
    """Return a chirp signal using the given parameters

    :param duration: duration of a single chirp (in secs)
    :param fs: sampling frequency
    :param low: lower bound of chirp freq, defaults to 20
    :param high: upper bound of chirp freq, defaults to 20000
    :param silence_duration: duration of silence before the chirp signal (in secs), defaults to 0
    :param double: generate double chirp if True, defaults to False
    :return: ndarray of the chirp signal with delay
    """
    sample_times = np.linspace(0, duration, fs * duration)
    chirp = scipy.signal.chirp(sample_times, low, duration, high)
    silence = np.zeros(fs * silence_duration)

    if double:
        delayed_chirp = np.append(silence, np.tile(chirp, 2))
    else:
        delayed_chirp = np.append(silence, chirp)
    
    return delayed_chirp

