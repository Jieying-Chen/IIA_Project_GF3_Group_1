import numpy as np
from math import ceil, floor
import scipy.interpolate
from . import encode

def sub_width(fs, dft_length):
    """Return the width of subcarrier bins in freq domain"""

    return fs / dft_length

def subcarrier_shift_gaussian(symbols, dft_length, fs, low_freq, high_freq, sigma, bits_per_symbol, constellation='bpsk'):

    # Calculate the width of each bin
    bin = sub_width(fs, dft_length)
    low_idx = ceil(low_freq / bin)
    high_idx = floor(high_freq / bin)
    # Used subcarriers
    subs_per_block = high_idx - low_idx + 1
    block_num = ceil(symbols.size / subs_per_block)
    if constellation == 'qpsk':
        symbols_padded = np.append(symbols, encode.qpsk_encode(np.zeros(bits_per_symbol * (block_num * subs_per_block - symbols.size))))
    elif constellation == 'bpsk':
        symbols_padded = np.append(symbols, encode.bpsk_encode(np.zeros(bits_per_symbol * (block_num * subs_per_block - symbols.size))))
        
    # Total subcarriers
    bin_num = int(dft_length / 2 - 1)
    output = np.array([])
    for row in np.reshape(symbols_padded, (-1, subs_per_block)):
        pre = np.random.normal(0, sigma, low_idx - 1) + 1j * np.random.normal(0, sigma, low_idx - 1)
        post = np.random.normal(0, sigma, bin_num - high_idx) + 1j * np.random.normal(0, sigma, bin_num - high_idx)
        output = np.concatenate((output, pre, row, post))

    return output

def subcarrier_extract(fft, dft_length, fs, low_freq, high_freq):

    bin = sub_width(fs, dft_length)
    low_idx = ceil(low_freq / bin)
    high_idx = floor(high_freq / bin)
    spb = int(dft_length / 2 - 1)
    output = np.array([])
    for row in np.reshape(fft, (-1, spb)):
        output = np.append(output, row[low_idx - 1:high_idx])

    return output

def symbols_to_ofdm(symbols, dft_length, cp_length):

    spb = int(dft_length / 2 - 1)
    reshaped = np.reshape(symbols, (-1, spb))
    x = np.array([])
    for row in reshaped:
        first_half = np.append(0, row)
        second_half = np.append(0, np.conjugate(row)[::-1])
        idft = np.fft.ifft(np.append(first_half, second_half))
        x = np.append(x, np.append(idft[-cp_length:], idft))

    return x

def ofdm_to_fourier(synced_ofdm, dft_length, cp_length):

    spb = int(dft_length / 2 - 1)
    datapoints_per_block = dft_length + cp_length
    reshaped = np.reshape(synced_ofdm, (-1, datapoints_per_block))
    output = np.array([])
    for row in reshaped:
        fft = np.fft.fft(row[cp_length:])
        output = np.append(output, fft[1:spb + 1])
    
    return output

def deconvolve(fft, h, dft_length, fs, low_freq, high_freq):

    bin = sub_width(fs, dft_length)
    low_idx = ceil(low_freq / bin)
    high_idx = floor(high_freq / bin)

    spb = 1 + high_idx - low_idx 
    idx_range = np.arange(low_idx, high_idx + 1)
    if h.size > 1:
        H_complex = np.fft.fft(h, n=dft_length)[idx_range]
    else:
        H_complex = np.tile(np.fft.fft(h), spb)
    output = np.array([])
    for row in np.reshape(fft, (-1, spb)):
        output = np.append(output, np.divide(row, H_complex))
    
    import matplotlib.pyplot as plt
    plt.plot(np.angle(output[:50], deg=True))
    plt.show()

    return output
