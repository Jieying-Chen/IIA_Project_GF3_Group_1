import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

#convolute the chirp with a reversed chirp, return max response position as the end of synchronising signal

def chirp_synchronize(input,fs= 44100, duration= 1):
    sample_times = np.linspace(0, duration, fs * duration)
    chirp = 0.1 * scipy.signal.chirp(sample_times, 20, duration, 20000)
    chirp_reverse = chirp[::-1]
    convolved = np.convolve(input, chirp_reverse)
    #plt.plot(abs(convoluted))
    #plt.show()
    end = np.argmax(abs(convolved))
    start = end - duration*fs
    return start,convolved
