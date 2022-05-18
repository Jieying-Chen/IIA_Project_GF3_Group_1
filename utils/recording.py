import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd

def record(duration,fs):
    """
    record the ambient sound for a certain duration with sampling frequency = fs

    :param duration: recording length
    :param fs: sampling frequency
    :return: the recording
    """
    recording = sd.rec(duration * fs, samplerate=fs, channels=1, blocking=True).flatten()
    plt.plot(recording)
    plt.show()
    return recording


