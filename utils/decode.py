import numpy as np
from utils import ldpc

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

def channel_noise(H,received_known,dft_length):
    received_known = np.reshape(received_known, (-1, dft_length))
    received_fft = np.fft.fft(received_known,axis = 1)
    print(received_fft.shape)
    noise = np.divide(received_fft,H)   #a matrix of complex noise
    print(noise.shape)
    sigma = np.std(np.reshape(noise,(-1)))
    return sigma


def ldpc_decode(deconvolved, H, sigma,spb):     #takes channel estimation at info subcarriers only
    c = ldpc.code()
    L = np.array([])
    for row in np.reshape(deconvolved,(-1,spb)):
        L1 = H*np.conj(H)* np.sqrt(2)*np.imag(row)/sigma
        L2 = H*np.conj(H)* np.sqrt(2)*np.real(row)/sigma
        L_stacked  = np.vstack((L1,L2)).reshape((-1,),order='F')
        L = np.append(L,L_stacked)

    #decode for each c.Nv bits
    decoded = np.array([])
    iteration = np.array([])
    for row in np.reshape(L,(-1,c.Nv)):
        app,iter = c.decode(row)
        decoded = np.append(decoded,app[:c.K])
        iteration = np.append(iteration,iter)

    return decoded,iteration
