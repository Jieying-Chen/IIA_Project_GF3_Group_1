import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
from scipy.io.wavfile import write

silence_duration = 1
chirp_duration = 1
fs = 48000 #sample freq

double_chirp = True

silence = np.zeros(fs * silence_duration)

#create chirp from 20Hz to 10kHz
sample_times = np.linspace(0, chirp_duration, fs * chirp_duration)
chirp = 0.1 * scipy.signal.chirp(sample_times, 20, chirp_duration, 20000)

header = np.concatenate((silence,chirp))

if double_chirp:
    header = np.concatenate((header,chirp))
    headername = 'header_doublechirp'
else:
    headername = 'header_singlechirp'

plt.plot(header)
plt.show()

write("{}.wav".format(headername), fs, header.astype(np.float32))