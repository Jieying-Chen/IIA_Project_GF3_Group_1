import numpy as np
import scipy.signal

def buffer(x, wlen, p):
    '''
    Parameters
    ----------
    x: ndarray
        Signal array
    wlen: int
        Window length
    p: int
        Number of samples to overlap

    Returns
    -------
    (n,wlen) ndarray
        Buffer array 
    n int
        Number of windows
    '''
    #n = x.size // wlen + 1 #number of windows
    n = 0 #number of windows
    buffer = []
    i = 0
    while i + wlen <= x.size:
        buffer.append(x[i:min(i + wlen, x.size)])
        i += (wlen - p)
        n += 1
        
    return np.array(buffer,dtype=object), n

#print(buffer(np.arange(10),3,2))

#with a known input signal
def channelest1(x, y, xwin,ywin,overlap):
    hsize = ywin - xwin
    x_w, wnumx = buffer(x,xwin,overlap)
    y_w, wnumy = buffer(y,ywin,overlap + hsize)
    
    if wnumx != wnumy:
        print('mismatch')
        return
    
    H = np.zeros(xwin, dtype=complex)
    for i in range(wnumx):
        currentx = x_w[i]
        currenty = y_w[i]
        #extendx = np.zeros(currenty.size)
        #extendx[:currentx.size] = currentx 
        X = np.fft.fft(currentx)
        Y = np.fft.fft(currenty)
        Y = Y[:X.size]

        H += np.divide(Y,X)
        
    H = H / wnumx
    h_est = np.fft.ifft(H)[:hsize]

    return np.real(h_est)

#with input noise
def channelest2(x, y, xwin,ywin,overlap,sigma):
    hsize = ywin - xwin
    x_w, wnumx = buffer(x,xwin,overlap)
    y_w, wnumy = buffer(y,ywin,overlap + hsize)

    if wnumx != wnumy:
        print('mismatch')
        return
    
    for i in range(wnumx):
        currentx = x_w[i]
        currenty = y_w[i]
        
        extendx = np.zeros(currenty.size)
        extendx[:currentx.size] = currentx 

        _, Sxx = scipy.signal.csd(extendx,extendx)
        _, Sxy = scipy.signal.csd(extendx,currenty)
        if i == 0:
            H = Sxy / Sxx
        else:
            H += Sxy / Sxx

    H = H / wnumx
    h_est = np.fft.ifft(H)[:hsize]
    h_est = np.real(h_est)
    return h_est

#with chirp
def channelest3(x, y, fs, hsize):   #x is reversed chirp. hsize=channel response length
    
    w = scipy.signal.convolve(y,x)
    h_est = w[fs-1:fs-1+hsize]/240 #hard code!!
    
    return h_est
        