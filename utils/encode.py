from numbers import Real
import numpy as np
from math import ceil
from utils import ldpc

def qpsk_encode(bitarray):
    """Return a numpy array of QPSK symbols encoded from the input bit array"""

    if bitarray.size % 2 != 0:
        raise ValueError('The length of bit array should be an even number')
    imag = bitarray[0::2]
    real = bitarray[1::2]
    output = 1 - 2 * real + 1j * (1 - 2 * imag)
    return output / np.sqrt(2)

def bpsk_encode(bitarray):

    output = -1 + 2 * bitarray

    return output

def zeros_padding(bitarray, dft_length, bits_per_symbol):
    """Return the input bit array padded with 0s so that the resulting bit array
    can be converted to a sequence of symbols whose size is divisible by dft_length/2-1"""

    symbol_num = ceil(bitarray.size / bits_per_symbol)
    bitarray = np.append(bitarray, np.zeros(bits_per_symbol * symbol_num - bitarray.size))
    symbols_per_block = int(dft_length / 2 - 1)
    block_num = ceil(symbol_num / symbols_per_block)
    bitarray = np.append(bitarray, np.zeros(bits_per_symbol * symbols_per_block * block_num - bitarray.size))

    return bitarray

def ldpc_encode(info):
    c = ldpc.code()

    #pad with zero such that info is integer times of c.K
    #pad = np.zeros(c.K - info.size%c.K)

    pad_num = c.K - info.size%c.K
    np.random.seed(8)
    pad = np.random.randint(2, size=int(pad_num))

    info = np.concatenate((info,pad))
    encoded = np.array([c.encode(info[i*c.K:(i+1)*c.K]) for i in range(info.size//c.K)]).reshape((-1))

    return encoded