from __future__ import print_function, division
from sklearn.metrics import confusion_matrix
from scipy.linalg import eig
from scipy import sqrt
import numpy as np
import pickle
import socket
import logging
import os

OVTK_StimulationLabel_Base = 0x00008100
commands = ['1', '2', '3', '4', '5', '6', '7', '8', '9']

# calculate score for each stimulus and select the target
def select_target(predictions, events):

    scores = []
    array = np.array(events)
    values = set(array)

    for i in range(1, len(values) + 1):
        item_index = np.where(array == i)
        cl_item_output = np.array(predictions)[item_index]          
        score = np.sum(cl_item_output == 0) / len(cl_item_output)
        scores.append(score)
        
        # check if the target returned by classifier is out of the commands set
        # or all stimuli were classified as non-target
    if scores.count(0) == len(scores):
    #    print self.scores
        feedback_data = '#'
    else:
       feedback_data = commands[scores.index(max(scores))]

    return feedback_data

# Filter
def eeg_filter(eeg, fs, low_pass, high_pass, order):
    B,A = sig.butter(order, np.array([low_pass,high_pass],dtype=float)/(fs/2), btype='bandpass')
    return sig.filtfilt(B, A, eeg, axis=0)

# Epoch
def eeg_epoch(eeg, epoch_length, markers):
    channels = int(eeg.shape[1])
    epoch_length = np.around(epoch_length)
    dur = np.arange(epoch_length[0], epoch_length[1]+1).reshape((epoch_length[1]+1,1)) * np.ones( (1, len(markers)),dtype=int)
    samples = len(dur)
    epoch_idx = dur + markers
    eeg_epochs = np.array(eeg[epoch_idx,:]).reshape((samples, len(markers), channels)).transpose((0,2,1))
    return eeg_epochs
    
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

# CCA
def cca(X,Y):
    if X.shape[1] != Y.shape[1]:
        raise Exception('unable to apply CCA, X and Y have different dimensions')
    z = np.vstack((X,Y))
    C = np.cov(z)
    sx = X.shape[0]
    sy = Y.shape[0]
    Cxx = C[0:sx, 0:sx] + 10**(-8)*np.eye(sx)
    Cxy = C[0:sx, sx:sx+sy]
    Cyx = Cxy.transpose()
    Cyy = C[sx:sx+sy, sx:sx+sy] + 10**(-8)*np.eye(sy)
    invCyy = np.linalg.pinv(Cyy)
    invCxx = np.linalg.pinv(Cxx)
    r, Wx = eig(invCxx.dot(Cxy).dot(invCyy).dot(Cyx))
    r = sqrt(np.real(r))
    r = np.sort(np.real(r),  axis=None)
    r = np.flipud(r)
    return r

def apply_cca(X,Y):
    coefs = []
    for i in range(Y.shape[0]):
        coefs.append(cca(X,Y[i,:,:]))
    coefs = np.array(coefs).transpose()
    return coefs

def predict(scores):
    return np.argmax(scores[0,:])

class HybridOnline(OVBox):

    def __init__(self):
        super(HybridOnline, self).__init__()
        self.experiment_state = 0   
        self.fs = 512
        self.channels = 0
        # self.signal = []
        self.signal =np.array([])
        self.tmp_list = []
        #
        self.erp_stims = []
        self.erp_stims_time = []
        self.erp_x = []
        self.erp_y = []
        self.erp_lowPass = 1
        self.erp_highPass = 10
        self.erp_filterOrder = 2
        self.erp_model = None
        self.erp_downSample = 4
        self.erp_movingAverage = 12
        self.erp_epochDuration = 0.7
        self.erp_samples = 0
        self.erp_channels = []
        self.erp_begin = 0
        self.erp_end = 0
        self.erp_model_path = []
        #
        self.ssvep_x = []
        self.ssvep_y = []
        self.ssvep_stims_time = []
        self.ssvep_lowPass = 5
        self.ssvep_highPass = 50
        self.ssvep_filterOrder = 6   
        self.ssvep_n_harmonics = 2
        self.ssvep_frequencies = ['idle', 14, 12, 10, 8]
        self.ssvep_epochDuration = 4.0
        self.ssvep_samples = 0
        self.ssvep_references = []   
        self.ssvep_channels = []   
        self.ssvep_mode = 'sync'  
        self.ssvep_begin = 0 
        #
        self.ends = 0
        self.nChunks = 0
        self.chunk = 0
        self.switch = False
        self.do_save = False 
        self.stream_signal = False  
        #
        self.feedback_data = 0
        self.hostname = '127.0.0.1'
        self.feedback_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.feedback_port = 12345         

    def initialize(self):
        self.fs = int(self.setting["Sample Rate"])
        #
        self.erp_lowPass = int(self.setting["ERP Low Pass"])
        self.erp_highPass = int(self.setting["ERP High Pass"])
        self.erp_filterOrder = int(self.setting["ERP Filter Order"])
        self.erp_downSample = int(self.setting["Downsample Factor"])
        self.erp_epochDuration = np.ceil(float(self.setting["ERP Epoch Duration (in sec)"]) * self.fs).astype(int)
        self.erp_movingAverage = int(self.setting["ERP Moving Average"])
        self.erp_model_path = self.setting["ERP Classifier"]
        self.erp_model = pickle.load(open(self.erp_model_path, 'rb'))
        #
        self.ssvep_lowPass = int(self.setting["SSVEP Low Pass"])
        self.ssvep_highPass = int(self.setting["SSVEP High Pass"])
        self.ssvep_filterOrder = int(self.setting["SSVEP Filter Order"])
        self.ssvep_epochDuration = float(self.setting["SSVEP Epoch Duration (in sec)"]) 
        self.ssvep_n_harmonics = int(self.setting["SSVEP Harmonics"])
        self.ssvep_samples = int(self.ssvep_epochDuration * self.fs)
        self.ssvep_mode = self.setting["SSVEP Mode"]
        t = np.arange(0.0, float(self.ssvep_samples)) / self.fs
        if self.ssvep_mode == 'sync':
            frequencies = self.ssvep_frequencies[1:]
        # generate reference signals
        x = [ [np.cos(2*np.pi*f*t*i),np.sin(2*np.pi*f*t*i)] for f in frequencies for i in range(1, self.ssvep_n_harmonics+1)]
        self.ssvep_references = np.array(x).reshape(len(frequencies), 2*self.ssvep_n_harmonics, self.ssvep_samples)
        

    def process(self):        
        
        # stream signal
        if self.input[0]:            
            signal_chunk = self.input[0].pop()

            if type(signal_chunk) == OVSignalHeader:
                self.channels, self.chunk = signal_chunk.dimensionSizes

            elif type(signal_chunk) == OVSignalBuffer:
                # print(self.getCurrentTime())
                # if(self.getCurrentTime() >= self.erp_begin): 
                if (self.stream_signal):                    
                    tmp = [signal_chunk[i:i+self.chunk] for i in range(0, len(signal_chunk), self.chunk)]
                    tmp = np.array(tmp)               
                    self.signal = np.hstack((self.signal, tmp)) if self.signal.size else tmp                
                    self.nChunks += 1                
                    del tmp          
            
            del signal_chunk
        
        
        # collect Stimulations markers and times for each paradigm
        if self.input[1]:
            chunk = self.input[1].pop()
            if type(chunk) == OVStimulationSet:
                for stimIdx in range(len(chunk)):
                    if chunk:
                        stim = chunk.pop()               
                        # print('Received Marker: ', stim.identifier, 'stamped at', stim.date, 's')
                        if(stim.identifier == OpenViBE_stimulation['OVTK_StimulationId_ExperimentStart']):
                            print('Experiment Start...')
                            # print('EXP start: ', stim.identifier, 'date:', stim.date)

                        # ERP session                        
                        if(stim.identifier == OpenViBE_stimulation['OVTK_StimulationId_TrialStart'] and not self.switch):                       
                            self.ssvep_y = []
                            self.ssvep_stims_time = []
                            self.tmp_list = []                            
                            if(len(self.erp_stims_time) == 0):
                                print('------ERP SESSION START----------')
                                self.stream_signal = True
                                self.erp_begin = stim.date
                                print('ERP begins at:', self.erp_begin)

                        '''
                        if (stim.identifier == OpenViBE_stimulation['OVTK_StimulationId_Target'] or 
                            stim.identifier == OpenViBE_stimulation['OVTK_StimulationId_NonTarget']) and not self.switch: 
                            self.erp_stims.append(stim.identifier - OpenViBE_stimulation['OVTK_StimulationId_Target'])
                            self.tmp_list.append(stim.date)
                        '''   
                        if (stim.identifier >= OVTK_StimulationLabel_Base) and (stim.identifier <= OpenViBE_stimulation['OVTK_StimulationId_LabelEnd']):
                            self.erp_stims.append(stim.identifier - OVTK_StimulationLabel_Base)
                            self.tmp_list.append(stim.date)        

                        if(stim.identifier == OpenViBE_stimulation['OVTK_StimulationId_TrialStop'] and not self.switch):                            
                            self.erp_stims_time.append(self.tmp_list)
                            print(self.erp_stims_time)
                            self.erp_end = stim.date
                            print('ERP begin at:', self.erp_begin)
                            print('ERP ends at:', self.erp_end)
                            # Classify ERP command
                            # send command as feedback
                            # switch to SSVEP
                            self.switch = True
                            self.stream_signal = False
                            print('Signal length ERP:', self.signal.shape)
                            self.signal = np.array([])
                            print('------ERP SESSION END----------')
                                                                        

                        # SSVEP session
                        if self.switch:
                            
                            self.erp_stims = []
                            self.erp_stims_time = []
                            if (stim.identifier == OpenViBE_stimulation['OVTK_StimulationId_TrialStart']):
                                print('------SSVEP SESSION START----------')
                                self.stream_signal = True
                                self.ssvep_begin = stim.date
                                print('SSVEP begins at:', self.ssvep_begin)

                            if stim.identifier > OVTK_StimulationLabel_Base and stim.identifier <= OVTK_StimulationLabel_Base+len(self.ssvep_frequencies):
                                self.ssvep_y.append(stim.identifier - OVTK_StimulationLabel_Base)
                                self.ssvep_stims_time.append(stim.date) 

                            if(stim.identifier == OpenViBE_stimulation['OVTK_StimulationId_TrialStop'] and self.ssvep_y):
                                # detect SSVEP Command
                                # send it as feedback
                                # switch back 
                                print('ssvep times ', self.ssvep_stims_time)
                                self.switch = False
                                self.stream_signal = False
                                print('Signal length SSVEP:', self.signal.shape)
                                self.signal = np.array([])
                                print('SSVEP begins at:', self.ssvep_begin)
                                print('SSVEP ends at:', stim.date)                                 
                                print('------SSVEP SESSION END----------')
                        
                        # Ending Experiment 
                        if stim.identifier == OpenViBE_stimulation['OVTK_StimulationId_ExperimentStop']:
                            print('EXPERIMENT ENDS ')
                            # if(self.ends == 0):
                            #     self.erp_end = stim.date
                            # else:
                            #     self.ssvep_end = stim.date
                            # self.ends += 1
                            # self.switch = True              
                        
                        # if self.ends == 2:
                        #    self.do_train = True
            del chunk
        return        

    def unintialize(self):
        pass


box = HybridOnline()
