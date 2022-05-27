import numpy as np

def qpsk_decode(input): # Map an narray of complex numbers to corresponding binary bits pairs
    imag = input.imag
    digit1 = imag < 0
    digit1 = digit1.astype(int)
    real = input.real
    digit2 = real < 0
    digit2 = digit2.astype(int)
    return np.concatenate(list(zip(digit1, digit2)))