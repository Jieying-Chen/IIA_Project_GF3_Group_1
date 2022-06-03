import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import pandas as pd
from utils import ofdm

#convolute the chirp with a reversed chirp, return max response position as the end of synchronising signal

def chirp_synchronize(input, chirp_range,fs= 48000, duration= 1,):
    sample_times = np.linspace(0, duration, fs * duration)
    chirp = 0.1 * scipy.signal.chirp(sample_times, chirp_range[0], duration, chirp_range[1])
    chirp_reverse = chirp[::-1]
    convolved = scipy.signal.convolve(input, chirp_reverse)
    #plt.plot(abs(convoluted))
    #plt.show()
    end = np.argmax(abs(convolved))
    start = end - duration*fs
    return start,convolved

def impulse_detect(signal,fs,duration, window_time=0.1, threshold = 3): #signal = convulved with reverse chirp
    noise = np.std(signal[int(duration*fs*1.2):])       #need to change appropriately
    window_size = window_time*fs
    signal = pd.Series(signal)
    std_rolling = signal.rolling(int(window_size)).std()
    ratio = std_rolling/noise
    event = []
    i=0
    while i<len(ratio):
        if ratio[i]>threshold:
            event.append(i)
            i += int(duration*fs*0.8) #mask how much behind an event?
        i+=1
    event_max_window = int(duration*fs*0.1/2)   #half window size
    event_max = [signal[event_time-event_max_window:event_time+event_max_window] for event_time in event]   #list of list of range to find
    event_max = [np.argmax(event_max[i])+event[i]-event_max_window for i in range(len(event))]
    return event_max

def phase_difference(received_signal, event,known_ofdm_data,CP_LENGTH,DFT_LENGTH,fs,low_freq,high_freq,repeat_time):
    #plot the phase difference between the estimation of channel with the two known ofdm symbol

    spb = ofdm.subcarriers_per_block(fs,DFT_LENGTH,low_freq,high_freq)
    received_ofdm_1 = received_signal[event[0]+48000:event[0]+48000+known_ofdm_data.size]
    received_ofdm_1 = received_ofdm_1[CP_LENGTH:]
    fft_1 = ofdm.ofdm_to_fourier(received_ofdm_1, DFT_LENGTH, cp_length=0) 
    discarded_1 = ofdm.subcarrier_extract(fft_1, DFT_LENGTH, fs, low_freq, high_freq)
    H1 = ofdm.known_ofdm_estimate(discarded_1,repeat_time,known_ofdm_data[CP_LENGTH:],DFT_LENGTH,low_freq,high_freq,fs)

    received_ofdm_2 = received_signal[event[1]-known_ofdm_data.size:event[1]]
    received_ofdm_2 = received_ofdm_2[CP_LENGTH:]
    fft_2 = ofdm.ofdm_to_fourier(received_ofdm_2, DFT_LENGTH, cp_length=0) 
    discarded_2 = ofdm.subcarrier_extract(fft_2, DFT_LENGTH, fs, low_freq, high_freq)
    H2 = ofdm.known_ofdm_estimate(discarded_2,repeat_time,known_ofdm_data[CP_LENGTH:],DFT_LENGTH,low_freq,high_freq,fs)

    phase_diff = np.angle(np.divide(H1,H2))
    plt.plot(phase_diff)
    plt.show()
    return phase_diff,H1,H2

