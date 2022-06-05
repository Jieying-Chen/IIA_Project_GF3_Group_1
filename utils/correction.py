from utils import ofdm
import matplotlib.pyplot as plt
import numpy as np
from math import ceil, floor
import scipy


def phase_correction(deconvolved, sample_shift, dft_length, fs, low_freq, high_freq):

    spb = ofdm.subcarriers_per_block(fs, dft_length , low_freq, high_freq)
    bin = ofdm.sub_width(fs, dft_length)
    low_idx = ceil(low_freq / bin)
    high_idx = floor(high_freq / bin)
    idx_range = np.arange(low_idx, high_idx + 1)
    omega_range = idx_range / dft_length * 2 * np.pi
    deconvolved = np.reshape(deconvolved, (-1, spb))
    output = np.array([])
    for i in range(deconvolved.shape[0]):
        to_correct = deconvolved[i]
        cumulative_shift = i / deconvolved.shape[0] * sample_shift
        multiplier = np.exp(-1j * omega_range * cumulative_shift)
        output = np.append(output, np.divide(to_correct, multiplier))

    return output
    
def phase_difference(received_signal, event,known_ofdm_data,CP_LENGTH,DFT_LENGTH,fs,low_freq,high_freq,repeat_time):
    #plot the phase difference between the estimation of channel with the two known ofdm symbol

    spb = ofdm.subcarriers_per_block(fs,DFT_LENGTH,low_freq,high_freq)
    received_ofdm_1 = received_signal[event[0]+48000:event[0]+48000+known_ofdm_data.size]
    H1 = ofdm.known_ofdm_estimate_edited(received_ofdm_1[CP_LENGTH:],known_ofdm_data[CP_LENGTH:CP_LENGTH+DFT_LENGTH],DFT_LENGTH,CP_LENGTH,low_freq,high_freq,fs)

    received_ofdm_2 = received_signal[event[1]-known_ofdm_data.size:event[1]]
    #received_ofdm_2 = received_signal[event[0]+671616-known_ofdm_data.size:event[0]+671616]
    H2 = ofdm.known_ofdm_estimate_edited(received_ofdm_2[CP_LENGTH:],known_ofdm_data[CP_LENGTH:CP_LENGTH+DFT_LENGTH],DFT_LENGTH,CP_LENGTH,low_freq,high_freq,fs)

    phase_diff = np.angle(np.divide(H1,H2))
    plt.plot(phase_diff)
    plt.title("phase difference of two channel estimation")
    plt.show()
    return phase_diff,H1,H2

def regression_correction(spb,slope1,intercept1,H1,H2,deconvolved,symbol_per_frame):
    x_2 = np.linspace(0,spb,num=spb)
    correction1 = np.exp(-1j*(slope1*x_2+intercept1))  #compensate with the regression result from last block
    phase_diff_1 = np.angle(np.divide(H1,H2)*correction1)
    slope2, intercept2, r_value, p_value, std_err = scipy.stats.linregress(x_2, phase_diff_1)
    plt.plot(phase_diff_1)
    plt.plot(slope2*x_2+intercept2)
    plt.title("2nd regression")
    plt.show()


    correction2 = np.exp(-(slope2*x_2+intercept2)*1j)
    phase_diff_3 = np.angle(np.divide(H1,H2)*correction1*correction2)
    plt.plot(phase_diff_3)
    plt.title("correctioned result")
    plt.show()
    slope = slope1+slope2
    intercept = intercept1+intercept2
    print(slope,intercept)

    #phase correction, given slope, intercept
    #assume slope and intercept increament linearly between all unknown ofdm symbols
    slopes = np.linspace(0,slope,num=symbol_per_frame)
    intercepts = np.linspace(0,intercept,num=symbol_per_frame)
    x_symbol = np.linspace(0,spb,num=spb)
    corrected = np.array([])
    deconvolved_reshape =  np.reshape(deconvolved,(-1,spb))
    #print(deconvolved_reshape.shape)
    for i in range(deconvolved_reshape.shape[0]):
        phase_correct = np.exp((x_symbol*slopes[i]+intercepts[i])*1j)
        corrected = np.append(corrected,deconvolved_reshape[i,:]*phase_correct)

    deconvolved = corrected
    return deconvolved


def phase_correction_edited(deconvolved, sample_shift, dft_length, cp_length, chirp_duration, fs, low_freq, high_freq,ori_length, repeat_time = 4):

    spb = ofdm.subcarriers_per_block(fs, dft_length , low_freq, high_freq)
    bin = ofdm.sub_width(fs, dft_length)
    low_idx = ceil(low_freq / bin)
    high_idx = floor(high_freq / bin)
    idx_range = np.arange(low_idx, high_idx + 1)
    omega_range = idx_range / dft_length * 2 * np.pi
    deconvolved = np.reshape(deconvolved, (-1, spb))
    output = np.array([])

    k = sample_shift / ori_length
    left_end = k * (chirp_duration*fs + cp_length + repeat_time * dft_length)
    right_end = k * (ori_length - (cp_length + repeat_time * dft_length))
    each_ofdm = (right_end - left_end) / deconvolved.shape[0] 

    
    for i in range(deconvolved.shape[0]):
        to_correct = deconvolved[i]
        cumulative_shift = i * each_ofdm + left_end * 0.8
        multiplier = np.exp(-1j * omega_range * cumulative_shift)
        output = np.append(output, np.divide(to_correct, multiplier))

    return output