from __future__ import division
# from numba import jit
# from scipy.linalg import eig
# from scipy import sqrt
from scipy.signal import butter, filtfilt 
import numpy as np
# utils
# import pickle
# import  os

# Filter
def eeg_filter(eeg, fs, low_pass, high_pass, order):
    # assert( high_pass > low_pass), 'Band-pass filtering: High cutoff frequency should be higher than low cutoff frequency'
    B, A = butter(order, np.array([low_pass, high_pass], dtype=float)/(fs/2), btype='bandpass')
    return filtfilt(B, A, eeg, axis=0)
    # return sig.filtfilt(B, A, eeg, axis=0, padtype='odd', padlen=3*(max(len(B),len(A))-1))
'''
# Epoch

def eeg_epoch(eeg, epoch_length, markers):
    channels = int(eeg.shape[1])
    epoch_length = np.around(epoch_length)
    dur = np.arange(epoch_length[0], epoch_length[1]).reshape((epoch_length[1],1)) * np.ones( (1, len(markers)),dtype=int)
    samples = len(dur)
    epoch_idx = dur + markers
    # eeg_epochs = np.array(eeg[epoch_idx,:]).reshape((samples, len(markers), channels), order='F').transpose((0,2,1))
    eeg_epochs = np.array(eeg[epoch_idx,:]).reshape((samples, len(markers), channels), order='F').transpose((0,2,1))
    return eeg_epochs
'''

# @jit(nopython=True)
def eeg_epoch(eeg, epoch_length, markers, fs):
    start = np.around(0.2*fs).astype(int)
    ep = epoch_length[0]
    epoch_length = [-0.2*fs, epoch_length[1]]
    channels = int(eeg.shape[1])
    epoch_length = np.around(epoch_length).astype(int)
    dur = np.arange(epoch_length[0], epoch_length[1]).reshape((epoch_length[1]-epoch_length[0],1)) * np.ones( (1, len(markers)),dtype=int)
    samples = len(dur)
    epoch_idx = dur + markers
    eeg_epochs = np.array(eeg[epoch_idx,:]).reshape((samples, len(markers), channels), order='F').transpose((0,2,1))
    baseline = eeg_epochs.mean(axis=0)
    eeg_epochs = eeg_epochs - baseline
    eeg_epochs = eeg_epochs[start+ep:, :, :]
    return eeg_epochs

# @jit(nopython=True)
# Feature extraction, downsample + moving average
def eeg_feature(eeg, downsample, moving_average):
    samples, channels, epochs, trials = eeg.shape
    x = eeg.reshape((samples, channels, epochs*trials))
    for tr in range(trials):
        for ch in range(channels):
            x[:,ch,tr] = np.convolve(x[:,ch,tr], np.ones(moving_average), 'same') / moving_average
    x = x[::downsample,:,:]
    samples, channels, trials = x.shape    
    return x.reshape((samples*channels,trials)).transpose((1,0))