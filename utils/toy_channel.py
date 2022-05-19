import numpy as np

def toy_channel(h,x,sigma):
    '''
    return the output of a noisy toy channel
    :param h: impulse response of the toy channel
    :param x: input signal
    :sigma: variance of the Gaussian noise
    :return: noisy output of the channel
    '''
    y = np.convolve(h,x)
    noise = sigma * np.random.randn(y.size)
    return y + noise