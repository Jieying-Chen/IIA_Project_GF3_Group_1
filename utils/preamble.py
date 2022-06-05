from time import time
from tracemalloc import start
import numpy as np
import scipy.signal
from math import ceil
try:
     from utils import ofdm,encode     #handles both file in utils folder and outside utils folder
except:
     import ofdm,encode

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
    silence = np.zeros(ceil(fs * silence_duration))

    if double:
        delayed_chirp = np.append(silence, np.tile(chirp, 2))
    else:
        delayed_chirp = np.append(silence, chirp)
    
    return delayed_chirp

def generate_known_ofdm(fs,dft_length,cp_length,low_freq,high_freq,encode_method,repeat_time, seed):
    if encode_method == 'bpsk':
        bits_per_symbol = 1
    elif encode_method == 'qpsk':
        bits_per_symbol = 2
    

    spb = ofdm.subcarriers_per_block(fs,dft_length,low_freq,high_freq)
    np.random.seed(seed)
    known_string = np.random.randint(2,size=2*spb)
    known_string_stack = np.tile(known_string,repeat_time-1)    #generate one ofdm symbol with prefix (cp length normal), the rest without (cp length set to 0)

    #convert string to complex symbols
    if encode_method == 'qpsk':
        symbols_first = encode.qpsk_encode(known_string)
        symbols_rest = encode.qpsk_encode(known_string_stack)
    elif encode_method == 'bpsk':
        symbols_first = encode.bpsk_encode(known_string)
        symbols_rest = encode.bpsk_encode(known_string_stack)

    #print(symbols_rest[::spb])  #first complex info qpsk

    #convert string of info to ofdm data
    known_shifted_first = ofdm.subcarrier_shift_gaussian(symbols_first, dft_length, fs, low_freq, high_freq, 0.01, bits_per_symbol, constellation=encode_method)
    known_ofdm_data_first = ofdm.symbols_to_ofdm(known_shifted_first, dft_length, cp_length)
    
    #known_shifted_rest = ofdm.subcarrier_shift_gaussian(symbols_rest, dft_length, fs, low_freq, high_freq, 0.01, bits_per_symbol, constellation=encode_method)
    known_shifted_rest = np.tile(known_shifted_first,repeat_time-1)
    known_ofdm_data_rest = ofdm.symbols_to_ofdm(known_shifted_rest, dft_length, cp_length=0)

    #combine the two parts
    known_ofdm_data = np.concatenate((known_ofdm_data_first,known_ofdm_data_rest))
    return known_ofdm_data,symbols_first

def transmission_start(fs,low_freq,high_freq,silence_duration):
    start_audio = generate_chirp(1, fs, low=low_freq, high=high_freq, silence_duration=silence_duration, double=False)
    return start_audio

def transmission_end(fs,low_freq,high_freq,silence_duration):
    chirp = generate_chirp(1, fs, low=low_freq, high=high_freq, silence_duration=0, double=False)
    silence = np.zeros(ceil(fs * silence_duration))
    end_audio = np.append(chirp,silence)
    return end_audio

def frame_assemble(chirp,known_ofdm,data):
    return np.concatenate((chirp,np.real(known_ofdm),np.real(data),np.real(known_ofdm),chirp))

def load_known_ofdm(CP_LENGTH = 512,repeat_time = 4):
    known_ofdm_symbol = np.load("known_ofdm_symbol.npy")
    time_domain = np.fft.ifft(known_ofdm_symbol)
    cyclic_prefix = time_domain[-CP_LENGTH:]
<<<<<<< HEAD
    stacked = np.tile(time_domain,repeat_time)
=======
    stacked = np.tile(time_domain, repeat_time)
>>>>>>> 70b78e1c5f39535cececee9b76a42cd46ea445df
    return np.append(cyclic_prefix,stacked)

if __name__ == "__main__":      #used for debugging functions, only run if running this file alone
    # known_ofdm = generate_known_ofdm(fs = 48000,dft_length=8192,cp_length=1024,low_freq=1000,high_freq=10000,encode_method='qpsk',repeat_time=5, seed=0)
    # a = known_ofdm[1024:]
    # spb = 1536
    # print(a[200::8192]) #should be the same
    import sounddevice as sd
    import matplotlib.pyplot as plt
    fs = 48000
    start_header = transmission_start(fs,1000,10000,1.2)
    plt.plot(start_header)
    plt.show()
    sd.play(start_header,fs,blocking=True)
    