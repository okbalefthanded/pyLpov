from __future__ import print_function, division
from sklearn.metrics import confusion_matrix
from scipy.linalg import eig
from scipy import sqrt
import numpy as np
import pickle

OVTK_StimulationLabel_Base = 0x00008100

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


class SSVEPpredictor(OVBox):

    def __init__(self):
        super(SSVEPpredictor, self).__init__()
        self.model = None
        self.predictions = []
        self.events = []
        self.trials_count = 0
        self.frequencies = ['idle', 14, 12, 10, 8]
        # self.frequencies = ['idle', 7.5, 8.57, 10, 12]
        self.num_harmonics = 0
        self.epoch_duration = 0
        self.fs = 512
        self.samples = 0
        self.references = []
        self.target_stimulations = []     


    def initialize(self):
        self.epoch_duration = float(self.setting['Epoch_duration'])
        self.num_harmonics = int(self.setting['Harmonics'])
        self.fs = float(self.setting['Sample_rate'])
        self.samples = int(self.epoch_duration * self.fs)
        t = np.arange(0.0, float(self.samples)) / self.fs
        if self.frequencies[0] == 'idle':
            frequencies = self.frequencies[1:]
        # generate reference signals
        x = [ [np.cos(2*np.pi*f*t*i),np.sin(2*np.pi*f*t*i)] for f in frequencies for i in range(1, self.num_harmonics+1)]
        # self.references = np.array(x).reshape(self.num_harmonics * len(frequencies), int(samples))
        self.references = np.array(x).reshape(len(frequencies), 2*self.num_harmonics, int(self.samples))  
        

    
    def process(self):
        # 
        if self.input[1]:
            chunk = self.input[1].pop()
            if type(chunk) == OVStimulationSet:
                for stimIdx in range(len(chunk)):
                    if chunk:
                        stim = chunk.pop()
                        print('Received Marker: ', stim.identifier, 'stamped at', stim.date, 's')

                        if stim.identifier == OpenViBE_stimulation['OVTK_StimulationId_TrialStop']:
                            self.trials_count += 1

                        if stim.identifier > OVTK_StimulationLabel_Base and stim.identifier <= OVTK_StimulationLabel_Base+len(self.frequencies):
                            self.target_stimulations.append(stim.identifier - OVTK_StimulationLabel_Base)
                            print("target is: ", self.frequencies[self.target_stimulations[-1]])

                        if stim.identifier == OpenViBE_stimulation['OVTK_StimulationId_ExperimentStop']:
                            # calculate % of correct detections
                            targets = np.array(self.target_stimulations)
                            predictions = np.array(self.predictions)
                            accuracy = (np.sum(targets == predictions) / len(self.target_stimulations)) * 100
                            print("Accuracy :", accuracy)
                            cm = confusion_matrix(targets, predictions)
                            print('Confusion matrix: ', cm)
                            # print("Targets: ", self.target_stimulations)
                            # print("Predictions: ", self.predictions)
                        

        if self.input[0]:
            buffer = self.input[0].pop()
            if type(buffer) == OVSignalBuffer:
                if (buffer):
                    channels = int(len(buffer) / self.samples)                    
                    epoch = np.array(buffer).reshape(channels, self.samples)                    
                    r = apply_cca(epoch, self.references)
                    command = predict(r) + 1 # temporarly, since we're using a sync mode
                    self.predictions.append(command)
                    print('Frequency detected %s Hz' %(self.frequencies[command]))
                    if command == self.target_stimulations[-1]:
                        print("Correct!")  
                    else:
                        print("Incorrect!")          



    
    def uninitialize(self):
        pass


box = SSVEPpredictor()