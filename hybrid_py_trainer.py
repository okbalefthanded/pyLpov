from __future__ import print_function, division
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import confusion_matrix
from swlda.swlda import SWLDA
from blda.blda import BLDA
from blda.feature_extraction import EPFL
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
    # clf = make_pipeline(preprocessing.StandardScaler(), LDA(solver='lsqr', shrinkage='auto')) # Shrinkage-LDA 
    # clf = make_pipeline(preprocessing.StandardScaler(), SWLDA()) # SWLDA
    # clf = make_pipeline(preprocessing.StandardScaler(), LogisticRegression(penalty='l1', solver='liblinear') ) # LogisticRegression + L1
    # clf = make_pipeline(preprocessing.StandardScaler(), GaussianProcessClassifier(kernel=1.0 * RBF(length_scale=1.0), optimizer=None) ) # Gaussian Process
    clf = make_pipeline(EPFL(decimation_factor=12, p=0.1), preprocessing.StandardScaler(), BLDA(verbose=True)) # BLDA 
    # cv = KFold(n_splits=5)
    # scores = cross_val_score(clf, x, y, cv=cv)
    # cv_results = cross_validate(clf, x, y, cv=cv, 
                                # scoring=('accuracy', 'roc_auc'),
                                # return_train_score=True)
    clf.fit(x, y)
    print("End training")
    #return clf, cv_results
    return clf

# Save model after training
def save_model(filename, model):
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

# ITCCA
def itcca(X, stims):
    stimuli_count = len(list(set(stims)))
    references = []
    for i in range(stimuli_count): 
        references.append(np.mean(X[:,:,np.where(stims==i+1)].squeeze(), axis=2))
    references = np.array(references).transpose((0,2,1))
    return references

def predict(scores):
    return np.argmax(scores[0,:])



class HybridClassifierTrainer(OVBox):

    def __init__(self):
        super(HybridClassifierTrainer, self).__init__()
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
        self.erp_epochDuration = np.ceil(float(self.setting["ERP Epoch Duration (in sec)"]) * self.fs).astype(int)
        self.erp_movingAverage = int(self.setting["ERP Moving Average"])
        self.erp_model_path = self.setting["ERP Classifier"]
        # self.erp_model = pickle.load(open(self.erp_model_path, 'rb'))
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
            self.ssvep_references = np.array([])
        

    def process(self):        
        
        # stream signal
        if self.input[0]:            
            signal_chunk = self.input[0].pop()

            if type(signal_chunk) == OVSignalHeader:
                self.channels, self.chunk = signal_chunk.dimensionSizes

            elif type(signal_chunk) == OVSignalBuffer: 
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

                            if (stim.identifier == OpenViBE_stimulation['OVTK_StimulationId_TrialStart'] and not self.ssvep_stims_time):
                                self.ssvep_begin = stim.date

                            if stim.identifier > OVTK_StimulationLabel_Base and stim.identifier <= OVTK_StimulationLabel_Base+len(self.ssvep_frequencies):
                                self.ssvep_y.append(stim.identifier - OVTK_StimulationLabel_Base)
                                self.ssvep_stims_time.append(stim.date) 
                        
                        # switching 
                        if stim.identifier == OpenViBE_stimulation['OVTK_StimulationId_ExperimentStop']:
                            if(self.ends == 0):
                                self.erp_end = stim.date
                            else:
                                self.ssvep_end = stim.date
                            self.ends += 1
                            self.switch = True                 
                        
                        if self.ends == 2:
                            self.do_train = True
            del chunk                     

        # Filter, Epoch and Train
        if self.do_train:
            # Reshape Signal
            print('Reshaping Signal')
            self.signal = self.signal.T         
            # 
            self.erp_stims_time = np.floor(np.array(self.erp_stims_time) * self.fs).astype(int)            
            self.ssvep_stims_time = np.floor(np.array(self.ssvep_stims_time) * self.fs).astype(int)
            self.erp_begin = int(np.floor(self.erp_begin * self.fs))
            self.erp_end = int(np.floor(self.erp_end * self.fs))
            self.ssvep_begin = int(np.floor(self.ssvep_begin * self.fs))
            self.ssvep_end = int(np.floor(self.ssvep_end * self.fs))
            self.ssvep_y = np.array(self.ssvep_y)
            print('ERP analysis...')
            erp_signal = self.signal[self.erp_begin:self.erp_end,:]
            # erp_signal = eeg_filter(erp_signal, self.fs, self.erp_lowPass, self.erp_highPass, self.erp_filterOrder)            
            # self.signal[self.erp_begin:self.erp_end,:] = erp_signal
            pickle.dump(self.signal, open('sig_calib_5_tr', 'wb'))
            pickle.dump(self.erp_stims_time, open('ev_calib_5_tr', 'wb'))
            erp_epochs = [] 
            for i in range(self.erp_stims_time.shape[0]):
                erp_epochs.append(eeg_epoch(self.signal, np.array([0, self.erp_epochDuration],dtype=int), self.erp_stims_time[i,:]))
            erp_epochs = np.array(erp_epochs).transpose((1,2,3,0))
            # self.erp_x = eeg_feature(erp_epochs, self.erp_downSample, self.erp_movingAverage)
            samples, channels, epochs, trials = erp_epochs.shape
            self.erp_x = erp_epochs.reshape((samples, channels, epochs*trials), order='F')
            
            self.erp_y = np.array(self.erp_stims, dtype=np.float32)
            self.erp_y[self.erp_y==1] = -1
            self.erp_y[self.erp_y==0] = 1            
            # pickle.dump(self.erp_x, open('erp_x', 'wb'))
            # pickle.dump(self.erp_y, open('erp_y', 'wb'))

            '''
            erp_model, scores = train(self.erp_x, self.erp_y) 
            
            print("Train Accuracy: %0.2f (+/- %0.2f)" % (scores['train_accuracy'].mean(), scores['train_accuracy'].std() * 2))
            print("Val Accuracy: %0.2f (+/- %0.2f)" % (scores['test_accuracy'].mean(), scores['test_accuracy'].std() * 2))
            print("Train ROC: %0.2f (+/- %0.2f)" % (scores['train_roc_auc'].mean(), scores['train_roc_auc'].std() * 2))
            print("Val ROC: %0.2f (+/- %0.2f)" % (scores['test_roc_auc'].mean(), scores['test_roc_auc'].std() * 2))
            '''


            del erp_signal
            del erp_epochs
            # SSVEP
            print('SSVEP analysis')
            ssvep_signal = self.signal[self.ssvep_begin:self.ssvep_end,:]
            ssvep_signal = eeg_filter(ssvep_signal, self.fs, self.ssvep_lowPass, self.ssvep_highPass, self.ssvep_filterOrder)            
            self.signal[self.ssvep_begin:self.ssvep_end,:] = ssvep_signal
            ssvep_epochs = eeg_epoch(self.signal, np.array([0, self.ssvep_samples],dtype=int), self.ssvep_stims_time)
            ssvep_predictions = []

            if self.ssvep_mode == 'sync':                
                sync_trials = np.where(self.ssvep_y != 1)
                ssvep_sync_epochs = ssvep_epochs[:,:,sync_trials].squeeze()                
                for i in range(ssvep_sync_epochs.shape[2]):
                    ssvep_predictions.append(predict(apply_cca(ssvep_sync_epochs[0:self.ssvep_samples,:,i].transpose((1,0)), self.ssvep_references)) + 1)
                ssvep_targets = np.array(self.ssvep_y[sync_trials]-1)
                
            elif self.ssvep_mode == 'async':                
                self.ssvep_references = itcca(ssvep_epochs, self.ssvep_y)
                ssvep_model = self.ssvep_references
                for i in range(ssvep_epochs.shape[2]):
                    ssvep_predictions.append(predict(apply_cca(ssvep_epochs[:,:,i].transpose((1,0)), self.ssvep_references)) + 1)
                ssvep_targets = np.array(self.ssvep_y)   

            ntrials = len(ssvep_predictions)
            ssvep_predictions = np.array(ssvep_predictions)
            accuracy = (np.sum(ssvep_targets == ssvep_predictions) / ntrials) * 100
            print("SSVEP Accuracy :", accuracy)
            cm = confusion_matrix(ssvep_targets, ssvep_predictions)
            print('SSVEP Confusion matrix:\n ', cm)
            del ssvep_signal
            del ssvep_epochs 
            self.do_train = False
            self.do_save = True

        if self.do_save:
            
            # erp_model = BLDA(verbose=False) 
            erp_model = train(self.erp_x, self.erp_y)           
            # erp_model.fit(self.erp_x, self.erp_y)
            save_model(self.erp_model_path, erp_model)
            #
            save_model(self.ssvep_model_path, ssvep_model)
            print("training ends...Saving...")
            self.do_save = False 
            stimSet = OVStimulationSet(0.,0.)    
            stimSet.append(OVStimulation(OpenViBE_stimulation['OVTK_StimulationId_ExperimentStop'], 0.,0.)) 
            self.output[0].append(stimSet)
            

    def uninitialize(self):
        self.signal = None
        self.erp_x = None
        self.erp_y = None
        self.ssvep_x = None
        self.ssvep_y = None
        self.ssvep_references = None
        self.erp_model = None     


box = HybridClassifierTrainer()