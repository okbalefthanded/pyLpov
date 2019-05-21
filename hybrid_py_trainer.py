from __future__ import print_function, division
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import confusion_matrix
from scipy.linalg import eig
from scipy import sqrt
import scipy.signal as sig
import numpy as np
import pickle
import os


OVTK_StimulationLabel_Base = 0x00008100


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

# Training function
def train(x, y):
    print("Training...")
    clf = make_pipeline(preprocessing.StandardScaler(), LDA(solver='lsqr', shrinkage='auto'))
    cv = KFold(n_splits=5)
    # scores = cross_val_score(clf, x, y, cv=cv)
    cv_results = cross_validate(clf, x, y, cv=cv, 
                                scoring=('accuracy', 'roc_auc'),
                                return_train_score=True)
    print("End training")
    return clf, cv_results

# Save model after training
def save_model(self, model):
        filename = self.setting['classifier_path']
        pickle.dump(model, open(filename, 'wb'))

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



class HybridClassifierTrainer(OVBox):

    def __init__(self):
        super(HybridClassifierTrainer, self).__init__()
        self.fs = 512
        self.channels = 0
        self.signal = []
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
        #
        self.ssvep_x = []
        self.ssvep_y = []
        self.ssvep_stims_time = []
        self.ssvep_lowPass = 5
        self.ssvep_highPass = 50
        self.ssvep_filterOrder = 6   
        self.ssvep_n_harmonics = 2
        self.ssvep_frequencies = ['idle', 14, 12, 10, 8]
        self.ssvep_epoch_duration = 4.0
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
        self.do_train = False
        self.do_save = False    
         

    def initialize(self):
        self.fs = int(self.setting["Sample Rate"])
        #
        self.erp_lowPass = int(self.setting["ERP Low Pass"])
        self.erp_highPass = int(self.setting["ERP High Pass"])
        self.erp_filterOrder = int(self.setting["ERP Filter Order"])
        self.erp_downSample = int(self.setting["Downsample Factor"])
        self.erp_epochDuration = float(self.setting["ERP Epoch Duration (in sec)"])
        self.erp_movingAverage = int(self.setting["ERP Moving Average"])
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
                self.signal.append(signal_chunk)
                self.nChunks += 1
       
       # collect Stimulations markers and times for each paradigm
        if self.input[1]:
            chunk = self.input[1].pop()
            if type(chunk) == OVStimulationSet:
                for stimIdx in range(len(chunk)):
                    if chunk:
                        stim = chunk.pop()               
                        print('Received Marker: ', stim.identifier, 'stamped at', stim.date, 's')
                        
                        # ERP session
                        if(stim.identifier == OpenViBE_stimulation['OVTK_StimulationId_TrialStart'] and not self.switch):
                            self.tmp_list = []                            
                            if(len(self.erp_stims_time) == 0):
                                self.erp_begin = stim.date
                            

                        if(stim.identifier == OpenViBE_stimulation['OVTK_StimulationId_TrialStop'] and not self.switch):                            
                            self.erp_stims_time.append(self.tmp_list)
                        
                        if (stim.identifier == OpenViBE_stimulation['OVTK_StimulationId_Target'] or 
                            stim.identifier == OpenViBE_stimulation['OVTK_StimulationId_NonTarget']) and not self.switch: 
                            self.erp_stims.append(stim.identifier - OpenViBE_stimulation['OVTK_StimulationId_Target'])
                            self.tmp_list.append(stim.date)                                                

                        # SSVEP session
                        if self.switch:
                            if stim.identifier > OVTK_StimulationLabel_Base and stim.identifier <= OVTK_StimulationLabel_Base+len(self.ssvep_frequencies):
                                self.ssvep_y.append(stim.identifier - OVTK_StimulationLabel_Base)
                                self.ssvep_stims_time.append(stim.date) 
                        
                        # switching 
                        if stim.identifier == OpenViBE_stimulation['OVTK_StimulationId_ExperimentStop']:
                            if(self.ends == 0):
                                self.erp_end = stim.date
                            self.ends += 1
                            self.switch = True                 
                        
                        if self.ends == 2:
                            self.do_train = True                        

        # Filter, Epoch and Train
        if self.do_train:
            # Reshape Signal
            self.signal = np.array(self.signal).reshape(self.channels, self.chunk*self.nChunks).T
            ## for dev purpose: saving the BOX object
            # fn = 'mrk'
            # print(self.erp_stims_time[0])
            # mrk = self.erp_stims_time
            # pickle.dump(mrk, open(fn, 'wb'))            
            self.erp_stims_time = np.floor(np.array(self.erp_stims_time) * self.fs).astype(int)            
            self.ssvep_stims_time = np.floor(np.array(self.ssvep_stims_time) * self.fs).astype(int)
            self.erp_begin = int(np.floor(self.erp_begin * self.fs))
            self.erp_end = int(np.floor(self.erp_end * self.fs))
            # fn = 'mr'
            # print(self.erp_stims_time[0])
            # mr = self.ssvep_stims_time
            # pickle.dump(mr, open(fn, 'wb')) 
            # Epoch ERP
            # sig = np.array(self.signal)            
            # filename = 'sig'            
            # pickle.dump(sig, open(filename, 'wb'))          
            # erp_signal = self.signal[self.erp_stims_time[0,0]:self.ssvep_stims_time[0],:]
            erp_signal = self.signal[self.erp_begin:self.erp_end,:]
            erp_signal = eeg_filter(erp_signal, self.fs, self.erp_lowPass, self.erp_highPass, self.erp_filterOrder)            
            self.signal[self.erp_begin:self.erp_end,:] = erp_signal
            erp_epochs = [] 
            for i in range(self.erp_stims_time.shape[0]):
                erp_epochs.append(eeg_epoch(self.signal, np.array([0, self.erp_epochDuration],dtype=int), self.erp_stims_time[i,:]))
            erp_epochs = np.array(erp_epochs).transpose((1,2,3,0))
            self.erp_x = eeg_feature(erp_epochs, self.erp_downSample, self.erp_movingAverage)
            self.erp_y = np.array(self.erp_stims, dtype=int)
            model, scores = train(self.x, self.y) 
            print("Train Accuracy: %0.2f (+/- %0.2f)" % (scores['train_accuracy'].mean(), scores['train_accuracy'].std() * 2))
            print("Val Accuracy: %0.2f (+/- %0.2f)" % (scores['test_accuracy'].mean(), scores['test_accuracy'].std() * 2))
            print("Train ROC: %0.2f (+/- %0.2f)" % (scores['train_roc_auc'].mean(), scores['train_roc_auc'].std() * 2))
            print("Val ROC: %0.2f (+/- %0.2f)" % (scores['test_roc_auc'].mean(), scores['test_roc_auc'].std() * 2))
            self.do_train = False
            self.do_save = True

        if self.do_save:
            print("training ends...Saving...")
            self.do_save = False 
            stimSet = OVStimulationSet(self.getCurrentTime(),
                                      self.getCurrentTime())    
            stimSet.append(OVStimulation(OpenViBE_stimulation['OVTK_StimulationId_ExperimentStop'], 
                                self.getCurrentTime(), 
                                self.getCurrentTime()))
            self.output[0].append(stimSet)
            

    def uninitialize(self):
        pass


box = HybridClassifierTrainer()