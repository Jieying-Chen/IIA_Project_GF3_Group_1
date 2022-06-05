from subprocess import STD_OUTPUT_HANDLE
from scipy.io import wavfile
import os.path
import sounddevice
import numpy as np
import matplotlib.pyplot as plt
import time

def signal_to_wav(s, fs, filename, path):
    """Write the signal s into path/filename.wav with sampling freq fs"""

    wavfile.write(os.path.join(path, "{}.wav".format(filename)), fs, s.astype(np.float32))

def wav_to_signal(filename, path):
    """Read data from path/filename.wav, return the signal in time domain and its sampling freq"""
    samplerate, data= wavfile.read(os.path.join(path, filename + ".wav"))

    return data, samplerate

def play_signal(s, fs, delay=0):
    """Play the signal s with sampling freq fs after delay seconds"""

    s = np.append(np.zeros(delay * fs), s)
    print("Playing after {} seconds delay...".format(delay))
    sounddevice.play(s, samplerate=fs, blocking=True)
    print("Finish playing")

def record(duration, fs):
    """Record and return the ambient sound for duration seconds with sampling freq fs"""

    print("Start recording for {} seconds...".format(duration))
    recording = sounddevice.rec(duration * fs, samplerate=fs, channels=1).flatten()
    key = input("Press Enter to stop recording")
    
    sounddevice.wait()

    
    print("Finish recording\n")
    plt.plot(recording)
    plt.show(block=False)

    return recording
