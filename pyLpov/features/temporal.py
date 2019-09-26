from __future__ import division
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class DownSample(BaseEstimator, TransformerMixin):

    def __init__(self, decimation_factor=12, moving_average=12):
        self.decimation_factor = decimation_factor
        self.moving_average = moving_average

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if X.ndim == 4:
            samples, channels, epochs, trials = X.shape
            X = X.reshape((samples, channels, epochs*trials))
        elif X.ndim == 3:
            samples, channels, trials = X.shape
        
        for tr in range(trials):
            for ch in range(channels):
                X[:,ch,tr] = np.convolve(X[:,ch,tr], np.ones(self.moving_average), 'same') / self.moving_average
        
        X = X[::self.decimation_factor,:,:]
        samples, channels, trials = X.shape    
        return X.reshape((samples*channels,trials)).transpose((1,0))