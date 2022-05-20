from scipy.io.wavfile import write
import os.path
import sounddevice
import numpy as np
import matplotlib.pyplot as plt

def signal_to_wav(s, fs, filename, path):
    """Write the signal s into path/filename.wav with sampling freq fs"""

    write(os.path.join(path, "{}.wav".format(filename)), fs, s)

def play_signal(s, fs, delay=0):
    """Play the signal s with sampling freq fs after delay seconds"""

    s = np.append(np.zeros(delay * fs), s)
    print("Playing after {} seconds delay...".format(delay))
    sounddevice.play(s, samplerate=fs, blocking=True)
    print("Finish playing")

def record(duration, fs):
    """Record and return the ambient sound for duration seconds with sampling freq fs"""

    print("Start recording for {} seconds...".format(duration))
    recording = sounddevice.rec(duration * fs, samplerate=fs, channels=1, blocking=True).flatten()
    print("Finish recording")
    plt.plot(recording)
    plt.show()

    return recording
