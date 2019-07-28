from __future__ import print_function, division
from sklearn.metrics import confusion_matrix
from scipy.linalg import eig
from scipy import sqrt
import scipy.signal as sig
import numpy as np
import socket
import logging
import pickle
import os
import gc

OVTK_StimulationLabel_Base = 0x00008100
commands = ['1', '2', '3', '4', '5', '6', '7', '8', '9']

# calculate score for each stimulus and select the target
def select_target(predictions, events):

    scores = []
    # print('events : ', events)
    array = np.array(events)
    values = set(array)

    for i in range(1, len(values) + 1):
        item_index = np.where(array == i)
        cl_item_output = np.array(predictions)[item_index]
           
        # score = np.sum(cl_item_output == 1) / len(cl_item_output)
        score = np.sum(cl_item_output) / len(cl_item_output)
        # score = np.sum(cl_item_output)
        scores.append(score)
        
    # check if the target returned by classifier is out of the commands set
    # or all stimuli were classified as non-target
    # print(' Predictions: ', predictions)
    # print(' Scores: ', scores)
    if scores.count(0) == len(scores):
    #    print self.scores
        feedback_data = '#'
    else:
       feedback_data = commands[scores.index(max(scores))]

    return feedback_data, scores.index(max(scores))

# Filter
def eeg_filter(eeg, fs, low_pass, high_pass, order):
    B,A = sig.butter(order, np.array([low_pass,high_pass],dtype=float)/(fs/2), btype='bandpass')
    return sig.filtfilt(B, A, eeg, axis=0)
    # return sig.filtfilt(B, A, eeg, axis=0, padtype='odd', padlen=3*(max(len(B),len(A))-1))

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

    
# Feature extraction, downsample + moving average
def eeg_feature(eeg, downsample, moving_average):
    samples, channels = eeg.shape[0], eeg.shape[1]
    if(eeg.ndim == 4):
        epochs, trials = eeg.shape[2], eeg.shape[3]
        x = eeg.reshape((samples, channels, epochs*trials))
    elif (eeg.ndim == 3):
        trials = eeg.shape[2]
        x = eeg
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

def itcca(X):
    stimuli_count = len(list(set(stims)))
    references = []
    for i in range(stimuli_count): 
        references.append(np.mean(X[:,:,np.where(stims==i+1)].squeeze(), axis=2))
    references = np.array(references).transpose((0,2,1))
    return references

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
        self.n_trials = 0
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
        self.erp_correct = 0
        self.erp_target = []
        self.erp_pred = []
        #
        self.ssvep_x = []
        self.ssvep_y = []
        self.ssvep_stims_time = []
        self.ssvep_model = None
        self.ssvep_model_path = []
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
        self.ssvep_correct = 0
        self.ssvep_target = []
        self.ssvep_pred = []
        #
        self.ends = 0
        self.nChunks = 0
        self.chunk = 0
        self.switch = False
        self.stream_signal = False  
        #
        self.feedback_data = 0
        self.hostname = '127.0.0.1'
        self.feedback_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.erp_feedback_port = 12345
        self.ssvep_feedback_port = 12346        

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
        self.ssvep_model_path = self.setting['SSVEP Classifier']
        t = np.arange(0.0, float(self.ssvep_samples)) / self.fs
        if self.ssvep_mode == 'sync':
            frequencies = self.ssvep_frequencies[1:]
            # generate reference signals
            x = [ [np.cos(2*np.pi*f*t*i),np.sin(2*np.pi*f*t*i)] for f in frequencies for i in range(1, self.ssvep_n_harmonics+1)]
            self.ssvep_references = np.array(x).reshape(len(frequencies), 2*self.ssvep_n_harmonics, self.ssvep_samples)
        elif self.ssvep_mode == 'async':
            self.ssvep_model = pickle.load(open(self.ssvep_model_path, 'rb'))
            # generate reference by itcca method
            self.ssvep_references = self.ssvep_model
        

    def process(self):        
        
        # stream signal
        if self.input[0]:            
            signal_chunk = self.input[0].pop()

            if type(signal_chunk) == OVSignalHeader:
                self.channels, self.chunk = signal_chunk.dimensionSizes

            elif type(signal_chunk) == OVSignalBuffer:
                # print(self.getCurrentTime())
                # if(self.getCurrentTime() >= self.erp_begin): 
                # if (self.stream_signal):                                       
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
                        # ERP session                        
                        if(stim.identifier == OpenViBE_stimulation['OVTK_StimulationId_TrialStart'] and not self.switch):                       
                            print('[ERP trial start]', stim.date)
                            self.ssvep_y = []
                            self.ssvep_stims_time = []
                            # self.tmp_list = []                            
                            if(len(self.erp_stims_time) == 0):
                                self.erp_begin = int(np.floor(stim.date * self.fs))                                
                        
                        if (stim.identifier == OpenViBE_stimulation['OVTK_StimulationId_Target'] or 
                            stim.identifier == OpenViBE_stimulation['OVTK_StimulationId_NonTarget']) and not self.switch: 
                            self.erp_y.append(stim.identifier - OpenViBE_stimulation['OVTK_StimulationId_Target'])
                         
                        if (stim.identifier >= OVTK_StimulationLabel_Base) and (stim.identifier <= OpenViBE_stimulation['OVTK_StimulationId_LabelEnd'] and not self.switch) :
                            self.erp_stims.append(stim.identifier - OVTK_StimulationLabel_Base) 
                            # print('[ERP stim]', stim.date, self.erp_stims[-1])
                            self.tmp_list.append(np.floor(stim.date*self.fs))        

                        if(stim.identifier == OpenViBE_stimulation['OVTK_StimulationId_TrialStop'] and not self.switch):                            
                            print('[ERP trial stop]', stim.date)
                            self.erp_stims_time = self.tmp_list 
                            self.erp_stims = np.array(self.erp_stims)                          
                            self.erp_end = int(np.floor(stim.date * self.fs)) 
                            self.erp_y = np.array(self.erp_y)                                                     
                            mrk = np.array(self.erp_stims_time).astype(int) - self.erp_begin                          
                            erp_signal = eeg_filter(self.signal[:, self.erp_begin:self.erp_end].T, self.fs, self.erp_lowPass, self.erp_highPass, self.erp_filterOrder)                            
                            erp_epochs = eeg_epoch(erp_signal, np.array([0, self.erp_epochDuration],dtype=int), mrk)
                            self.erp_x = erp_epochs
                            # print('ERP shape: ', self.erp_x.shape)
                            # self.erp_x = eeg_feature(erp_epochs, self.erp_downSample, self.erp_movingAverage)
                            
                            predictions = self.erp_model.predict(self.erp_x)
                            # predictions = self.erp_model.decision_function(self.erp_x)
                            self.command, idx = select_target(predictions, self.erp_stims)                        
                            print('[ERP] Command to send is: ', self.command)
                           
                            self.feedback_socket.sendto(self.command, (self.hostname, self.erp_feedback_port))                       
                            # switch to SSVEP, free memory
                            self.erp_y[self.erp_y == 1] = -1
                            self.erp_y[self.erp_y == 0] = 1
                            # tg = 0
                            # print('shape : ', self.erp_stims.shape)
                            # print('real classes: ', self.erp_y)
                            self.erp_pred.append(self.command)
                            tg = np.where(self.erp_y == 1)
                            # print('tg : ', tg)                                
                            print('[ERP Target] : ', self.erp_stims[tg[0][0]] )
                            self.erp_target.append(self.erp_stims[tg[0][0]]) 
                            if self.command == '#':
                                print('NO ERP detection ...')
                            elif int(self.command) == self.erp_stims[tg[0][0]]:                               
                                print('[ERP Correct! ]')
                                self.erp_correct += 1
                        
                            self.erp_x = []
                            del erp_signal
                            del erp_epochs
                            del mrk
                            self.tmp_list = []
                            self.switch = True                                                                                                  

                        # SSVEP session
                        if self.switch:
                                                    
                            self.erp_stims = []
                            self.erp_stims_time = []
                            self.erp_y = []
                            if (stim.identifier == OpenViBE_stimulation['OVTK_StimulationId_TrialStart']):                                
                                # self.stream_signal = True
                                print('[SSVEP trial start]', stim.date)
                                self.ssvep_begin = int(np.floor(stim.date * self.fs))                                

                            if stim.identifier >= OVTK_StimulationLabel_Base and stim.identifier <= OVTK_StimulationLabel_Base+len(self.ssvep_frequencies):
                                self.ssvep_y.append(stim.identifier - OVTK_StimulationLabel_Base)
                                # print('[SSVEP stim]', stim.date, self.ssvep_y[-1])
                                self.ssvep_stims_time.append(np.floor(stim.date*self.fs)) 

                            if(stim.identifier == OpenViBE_stimulation['OVTK_StimulationId_TrialStop'] and self.ssvep_y):
                                print('[SSVEP trial stop]', stim.date)                               
                                
                                self.ssvep_end = int(np.floor(stim.date*self.fs))                                
                                # SSVEP                                
                                self.ssvep_stims_time = np.array(self.ssvep_stims_time).astype(int) - self.ssvep_begin                                                             
                                ssvep_signal = eeg_filter(self.signal[:,self.ssvep_begin:self.ssvep_end].T, self.fs, self.ssvep_lowPass, self.ssvep_highPass, self.ssvep_filterOrder)   
                                ssvep_epochs = eeg_epoch(ssvep_signal, np.array([0, self.ssvep_samples],dtype=int), self.ssvep_stims_time).squeeze()                                
                                # ssvep_predictions = predict(apply_cca(ssvep_epochs.transpose((1,0)), self.ssvep_references[:,:,0:self.ssvep_samples])) + 1
                                ssvep_epochs = ssvep_epochs.transpose((2,1,0)) 
                                print('SSVEP epochs : shape : ', ssvep_epochs.shape)
                                ssvep_predictions = self.ssvep_model.predict(ssvep_epochs)
                                self.command = ssvep_predictions
                                print('[SSVEP] preds:', ssvep_predictions, ' target:', self.ssvep_y)
                                if ssvep_predictions == self.ssvep_y:
                                    self.ssvep_correct += 1
                                print('[SSVEP] Sending as feedback: ', self.command)                                                               
                                self.feedback_socket.sendto(str(self.command), (self.hostname, self.ssvep_feedback_port))
                                self.ssvep_target.append(self.ssvep_y[-1])
                                self.ssvep_pred.append(self.command)
                                '''
                                if self.ssvep_mode == 'sync':                
                                    sync_trials = np.where(self.ssvep_y != 1)
                                    ssvep_sync_epochs = ssvep_epochs[:,:,sync_trials].squeeze()
                                    ssvep_predictions = []                                    
                                    ssvep_predictions.append(predict(apply_cca(ssvep_sync_epochs[0:self.ssvep_samples,:].transpose((1,0)), self.ssvep_references)) + 1)
                                    
                                else:
                                    pass # TODO
            
                                print("SSVEP Accuracy :", accuracy)
                                cm = confusion_matrix(ssvep_targets, ssvep_predictions)
                                print('SSVEP Confusion matrix: ', cm)                              
                                '''       
                                # del ssvep_signal
                                # del ssvep_epochs        
                                self.switch = False
                                self.n_trials += 1
                                print('Trial N :', self.n_trials, ' / ERP Target : ', self.erp_target)
                                print('Trial N :', self.n_trials, ' / ERP Pred : ', self.erp_pred)
                                print('Trial N :', self.n_trials, ' / SSVEP Target : ', self.ssvep_target)
                                print('Trial N :', self.n_trials, ' / SSVEP Pred : ', self.ssvep_pred)
                                print('Trial N :', self.n_trials, ' / ERP Accuracy : ', (self.erp_correct / self.n_trials) * 100, '/  SSSVEP Accuracy : ', (self.ssvep_correct / self.n_trials) * 100 )
                                
                                # self.stream_signal = False                                
                                # self.signal = np.array([])                                                                 
                                                    
                        # Ending Experiment 
                        if stim.identifier == OpenViBE_stimulation['OVTK_StimulationId_ExperimentStop']:
                            print('EXPERIMENT ENDS')
                            print(' ERP Accuracy : ', (self.erp_correct / self.n_trials) * 100)
                            print(' SSSVEP Accuracy : ', (self.ssvep_correct / self.n_trials) * 100)                           
                            self.switch = False
                            del self.signal
                            del self.erp_x
                            del self.erp_model
                            del self.ssvep_x
                            del self.ssvep_model
                            stimSet = OVStimulationSet(0.,0.)    
                            stimSet.append(OVStimulation(OpenViBE_stimulation['OVTK_StimulationId_ExperimentStop'], 0.,0.)) 
                            self.output[0].append(stimSet)

            del chunk
        return        

    def unintialize(self):
        pass

box = HybridOnline()
