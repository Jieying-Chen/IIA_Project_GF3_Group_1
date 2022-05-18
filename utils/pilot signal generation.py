import sounddevice as sd
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal


#pilot signal generation

duration = 2
fs = 44100 #sample freq
sample_times = np.linspace(0, duration, fs * duration)


#create chirp from 20Hz to 10kHz
chirp = 0.1 * scipy.signal.chirp(sample_times, 20, duration, 10000) 

#impulse signal
impulse = scipy.signal.unit_impulse(fs * duration) 

#single frequency signal
single_freq = 441
single = 0.1 * scipy.signal.chirp(sample_times, single_freq, duration, single_freq)

#Zadoff-Chu sequence
def ZadoffChu(order, length, index=0):
    #length = length of sequence = duration * fs?
    #order = a parameter of the sequence, can be 1 (as in the eg)
    cf = length % 2
    n = np.arange(length)
    arg = np.pi * order * n * (n+cf+2*index)/length
    return np.exp(-1j*arg)


x = np.linspace(0,44100,fs * duration)

#plt.plot(chirp)
plt.plot(x,np.fft.fft(impulse))
plt.show()
#sd.play(chirp,samplerate=fs, blocking=True)