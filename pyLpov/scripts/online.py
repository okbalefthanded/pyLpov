'''
Base class for Online processing scripts
'''
from sklearn.metrics import confusion_matrix
import numpy as np
import socket
import logging
import pickle
import datetime
import random
import abc
import os
import gc

class OnLine(object):

    def __init__(self, paradigm=None):
        self.paradigm = paradigm
        self.experiment_mode = 'copy'
        self.signal = np.array([])
        self.channels = 0
        self.tmp_list = []
        self.n_trials = 0
        self.model = None
        self.model_path = None
        self.deep_model = None
        self.model_file_type = None
        #
        self.stims = []
        self.stims_time = []
        self.x = []
        self.y = [] 
        #
        self.fs = 512
        self.samples = 0
        self.low_pass = 4
        self.high_pass = 60
        self.filter_order = 2
        self.epoch_duration = 1.0
        self.begin = 0
        self.end = 0
        self.correct = 0
        self.target = []
        self.pred = []
        #
        self.ends = 0
        self.nChunks = 0
        self.chunk = 0
        #
        self.command = None
        self.hostname = '127.0.0.1'
        self.feedback_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.feedback_port = 12345   
        self.do_feedback = False 
        self.trial_ended = False
        # 
        self.dur  = 0
        self.tr_dur = []
    
    def stream_signal(self, signal_chunk):
        signal_chunk = signal_chunk.pop()

        if signal_chunk.__class__.__name__ == 'OVSignalHeader':
                self.channels, self.chunk = signal_chunk.dimensionSizes

        elif signal_chunk.__class__.__name__ == 'OVSignalBuffer':                                               
                tmp = [signal_chunk[i:i+self.chunk] for i in range(0, len(signal_chunk), self.chunk)]
                tmp = np.array(tmp)               
                self.signal = np.hstack((self.signal, tmp)) if self.signal.size else tmp                
                self.nChunks += 1                
                del tmp          
            
        # del signal_chunk

    @abc.abstractmethod
    def filter_and_epoch(self, stim):
        pass    

    @abc.abstractmethod
    def predict(self):
        pass

    @abc.abstractmethod
    def print_if_target(self):
        pass
    
    def init_data(self):
        self.x = []
        self.y = []
        self.tmp_list = []
        self.stims = []
        self.stims_time = []                     
        self.n_trials += 1

    def print_results(self):
        '''
        '''                
        print(f"Trial N: {self.n_trials}  / {self.paradigm} Target: {self.target}")
        print(f"Trial N: {self.n_trials}  / {self.paradigm} Pred: {self.pred}")
        print(f"Trial N: {self.n_trials}  / {self.paradigm} Accuracy: {(self.correct / self.n_trials)*100}")  

    @abc.abstractmethod
    def experiment_end(self):
        pass

    def terminate(self):
        if self.experiment_mode == 'Copy':
            print('Accuracy : ', (self.correct / self.n_trials) * 100)
            cm = confusion_matrix(np.array(self.target), np.array(self.pred))
            print('Confusion matrix: ', cm)
        self.switch = False
        del self.signal
        del self.x
        del self.model
        jitter = np.diff(self.tr_dur)
        print('Trial durations delay: ',  jitter, jitter.min(), jitter.max(), jitter.mean())

    def predict_deep_model(self):
        '''
        '''
        self.transpose_epochs() 
        if self.model_file_type == 'h5':               
            predictions = self.model.predict(self.x)
        elif self.model_file_type == 'pth':                    
            predictions = self.model.predict(self.x, normalize=True)                    
        elif self.model_file_type == 'xml':
            predictions = [self.model.predict(self.x[i][None,...]) for i in range(self.x.shape[0])]
        return predictions

    def transpose_epochs(self):
        if self.x.ndim == 2:
            self.x = self.x[...,None]
        self.x = self.x.transpose((2,1,0))
