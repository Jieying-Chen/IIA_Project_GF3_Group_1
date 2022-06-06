import numpy as np
from utils import ldpc
from utils.ofdm import deconvolve
import numpy as np

def qpsk_decode(input): 
    """Map an narray of complex numbers to corresponding binary bits pairs"""

    imag = input.imag
    digit1 = imag < 0
    digit1 = digit1.astype(int)
    real = input.real
    digit2 = real < 0
    digit2 = digit2.astype(int)
    return np.concatenate(list(zip(digit1, digit2)))

def bpsk_decode(input):

    return (input > 0).astype(int)

def channel_noise(H,received_known,knonw_ofdm_data,cp_length, dft_length):
    received_known = np.reshape(received_known, (-1, dft_length))
    received_fft = np.fft.fft(received_known,axis = 1)

    known_fft = np.fft.fft(knonw_ofdm_data[cp_length:dft_length+cp_length])
    noise = np.divide(received_fft,H)- np.tile(known_fft,(4,1))  #a matrix of complex noise
    #print(noise.shape)
    noise = np.reshape(noise,(-1))
    #print(noise)
    sigma = np.std(np.concatenate((np.real(noise),np.imag(noise))))
    return sigma


def ldpc_decode(deconvolved,sigma,spb):     #takes channel estimation at info subcarriers only
    c = ldpc.code(standard = "802.16", rate = '1/2', z  = 64)
    L = np.array([])
    for row in np.reshape(deconvolved,(-1,spb)):
        L1 = np.sqrt(2)*np.imag(row)/sigma
        L2 = np.sqrt(2)*np.real(row)/sigma
        L_stacked  = np.vstack((L1,L2)).reshape((-1,),order='F')
        L = np.append(L,L_stacked)
    print(L.shape)
    #decode for each c.Nv bits
    decoded = np.array([])
    iteration = np.array([])
    remove = L.size%c.Nv
    for row in np.reshape(L[:L.size-remove],(-1,c.Nv)):
        #print(row[:30])
        app,iter = c.decode(row)
        decoded = np.append(decoded,app[:c.K])
        iteration = np.append(iteration,iter)

    return decoded,iteration


if __name__ == "__main__":

    deconvolved = np.load("deconvolved.npy")
    H_known_ofdm = np.load("H_known_ofdm")
    sigma = 1
    spb = 718
    decoded,iteration = decode.ldpc_decode(deconvolved,H_known_ofdm,sigma,spb)
    