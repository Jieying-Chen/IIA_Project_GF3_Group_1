import numpy as np
from math import ceil, floor
import matplotlib as plt
import scipy.interpolate
try:
     from utils import encode
except:                         #handles both file in utils folder and outside utils folder
     import encode

    
def sub_width(fs, dft_length):
    """Return the width of subcarrier bins in freq domain"""

    return fs / dft_length

def subcarrier_shift_gaussian(symbols, dft_length, fs, low_freq, high_freq, sigma, bits_per_symbol, constellation='qpsk'):
    #put constellation symbols into right bins and add gaussian noise for other bins up to fs/2

    # Calculate the width of each bin
    bin = sub_width(fs, dft_length)
    low_idx = ceil(low_freq / bin)
    high_idx = floor(high_freq / bin)
    # Number of subcarriers with information
    encoded_subs_per_block = high_idx - low_idx + 1
    block_num = ceil(symbols.size / encoded_subs_per_block)
    if constellation == 'qpsk':
        symbols_padded = np.append(symbols, encode.qpsk_encode(np.zeros(bits_per_symbol * (block_num * encoded_subs_per_block - symbols.size))))
    elif constellation == 'bpsk':
        symbols_padded = np.append(symbols, encode.bpsk_encode(np.zeros(bits_per_symbol * (block_num * encoded_subs_per_block - symbols.size))))
        
    # Total subcarriers to modulate
    total_subs_per_block = int(dft_length / 2 - 1)
    output = np.array([])
    for row in np.reshape(symbols_padded, (-1, encoded_subs_per_block)):
        pre = np.random.normal(0, sigma, low_idx - 1) + 1j * np.random.normal(0, sigma, low_idx - 1)
        post = np.random.normal(0, sigma, total_subs_per_block - high_idx) + 1j * np.random.normal(0, sigma, total_subs_per_block - high_idx)
        output = np.concatenate((output, pre, row, post))

    return output

def subcarrier_shift_ofdm(symbols, dft_length, fs, low_freq, high_freq, bits_per_symbol=2, seed=0):
    # put constellation symbols into right bins and add random ofdm symbols to the other bins up to fs/2

    # Calculate the width of each bin
    bin = sub_width(fs, dft_length)
    low_idx = ceil(low_freq / bin)
    high_idx = floor(high_freq / bin)

    # Number of subcarriers with information
    encoded_subs_per_block = high_idx - low_idx + 1
    block_num = ceil(symbols.size / encoded_subs_per_block)
    symbols_padded = np.append(symbols, encode.qpsk_encode(np.random.randint(0, 2, bits_per_symbol * (block_num * encoded_subs_per_block - symbols.size))))

    # Total subcarriers to modulate
    total_subs_per_block = int(dft_length / 2 - 1)
    output = np.array([])
    np.random.seed(seed)
    for row in np.reshape(symbols_padded, (-1, encoded_subs_per_block)):
        pre_phases = np.pi * (1 / 4 + 1 / 2 * np.random.randint(0, 4, low_idx - 1))
        pre = np.exp(1j * pre_phases)
        post_phases = np.pi * (1 / 4 + 1 / 2 * np.random.randint(0, 4, total_subs_per_block - high_idx))
        post = np.exp(1j * post_phases)
        output = np.concatenate((output, pre, row, post))

    return output

def subcarrier_extract(fft, dft_length, fs, low_freq, high_freq):

    bin = sub_width(fs, dft_length)
    low_idx = ceil(low_freq / bin)
    high_idx = floor(high_freq / bin)
    mod_subs_per_block = int(dft_length / 2 - 1)
    output = np.array([])
    for row in np.reshape(fft, (-1, mod_subs_per_block)):
        output = np.append(output, row[low_idx - 1:high_idx])

    return output

def symbols_to_ofdm(symbols, dft_length, cp_length):        #add zero, complex conjugate, cyclic prefix

    mod_subs_per_block = int(dft_length / 2 - 1)
    reshaped = np.reshape(symbols, (-1, mod_subs_per_block))
    x = np.array([])
    for row in reshaped:
        first_half = np.append(1, row)
        second_half = np.append(1, np.conjugate(row)[::-1])
        idft = np.fft.ifft(np.append(first_half, second_half))
        if cp_length==0:
            cyclic_prefix = np.array([])
        else:
            cyclic_prefix=idft[-cp_length:]
        x = np.append(x, np.append(cyclic_prefix, idft))

    return x

def ofdm_to_fourier(synced_ofdm, dft_length, cp_length):

    mod_subs_per_block = int(dft_length / 2 - 1)
    datapoints_per_block = dft_length + cp_length
    reshaped = np.reshape(synced_ofdm, (-1, datapoints_per_block))
    output = np.array([])
    for row in reshaped:
        fft = np.fft.fft(row[cp_length:])
        output = np.append(output, fft[1:mod_subs_per_block + 1])
    
    return output

def deconvolve(fft, H, dft_length, fs, low_freq, high_freq):

    bin = sub_width(fs, dft_length)
    low_idx = ceil(low_freq / bin)
    high_idx = floor(high_freq / bin)

    encoded_subs_per_block = high_idx - low_idx + 1
    idx_range = np.arange(low_idx, high_idx + 1)

    H_subcarrier = H[idx_range]

    #return fft-ed values with the channel freq response
    output = np.array([])
    for row in np.reshape(fft, (-1, encoded_subs_per_block)):
        output = np.append(output, np.divide(row, H_subcarrier))
    
    # import matplotlib.pyplot as plt
    # plt.plot(np.angle(output[:50], deg=True))
    # plt.show()

    return output

def subcarriers_per_block(fs,dft_length,low_freq,high_freq):
    bin = sub_width(fs, dft_length)
    low_idx = ceil(low_freq / bin)
    high_idx = floor(high_freq / bin)
    # Used subcarriers
    subs_per_block = high_idx - low_idx + 1
    return subs_per_block

def known_ofdm_estimate_edited(received_known, known_ofdm_data, dft_length, cp_length, low_freq, high_freq, fs):
    spb = subcarriers_per_block(fs, dft_length, low_freq, high_freq)
    avg_received_known = np.average(np.reshape(received_known, (-1, dft_length)), axis=0)
    received_fft = np.fft.fft(avg_received_known)
    known_fft = np.fft.fft(known_ofdm_data)
    bin = sub_width(fs, dft_length)
    low_idx = ceil(low_freq / bin)
    high_idx = floor(high_freq / bin)
    idx_range = np.arange(low_idx, high_idx + 1)

    H = np.divide(received_fft, known_fft)
    return H[idx_range], H
<<<<<<< HEAD

def unwrap_phase(phase):
    for i in range(1, len(phase)):
        if phase[i] - phase[i-1] > 1.5*np.pi:
            phase[i:] -= 2*np.pi
        elif phase[i-1] - phase[i] > 1.5*np.pi:
            phase[i:] += 2*np.pi          
    return phase
=======
>>>>>>> 71cb393a7645ffa85992eb6c13b9ab854e660525
