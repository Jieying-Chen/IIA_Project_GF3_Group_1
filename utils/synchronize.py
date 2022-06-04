import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import pandas as pd

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

