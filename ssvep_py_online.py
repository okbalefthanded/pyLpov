from __future__ import print_function, division
from sklearn.cross_decomposition import CCA
from scipy.linalg import eig
from scipy import sqrt
import numpy as np
import pickle
import socket

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
        self.frequencies = ['idle', 7, 9, 11, 13]
        self.num_harmonics = 0
        self.epoch_duration = 0
        self.fs = 512
        self.references = []
        self.command = None
        self.hostname = '127.0.0.1'
        self.feedback_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.feedback_port = 12345   
        self.do_feedback = False 


    def initialize(self):
        self.epoch_duration = float(self.setting['Epoch_duration'])
        self.num_harmonics = int(self.setting['Harmonics'])
        self.fs = float(self.setting['Sample_rate'])
        samples = self.epoch_duration * self.fs
        t = np.arange(0.0, samples) / self.fs
        if self.frequencies[0] == 'idle':
            frequencies = self.frequencies[1:]
        # generate reference signals
        x = [ [np.cos(2*np.pi*f*t*i),np.sin(2*np.pi*f*t*i)] for f in frequencies for i in range(1, self.num_harmonics+1)]
        self.references = np.array(x).reshape(len(frequencies), 2*self.num_harmonics, int(samples))  
        

    
    def process(self):
        # 
        if self.input[1]:
            chunk = self.input[1].pop()
            if type(chunk) == OVStimulationSet:
                for stimIdx in range(len(chunk)):
                    if chunk:
                        stim = chunk.pop()
                        print('Received Marker: ', stim.identifier, 'stamped at', stim.date, 's')

                        if stim.identifier ==  OpenViBE_stimulation['OVTK_StimulationId_TrialStart']:
                            print('Trial Start...')
                            self.command = None                             
                            self.trials_count += 1
                        
                        if stim.identifier == OpenViBE_stimulation['OVTK_StimulationId_TrialStop']:
                            print('Trial Stop... sending feedback')
                            # do feedback
                            if self.do_feedback:
                                print('Sending as feedback: ', self.command)
                                self.feedback_socket.sendto(self.command, (self.hostname, self.feedback_port))
                                self.do_feedback = False
                          

        if self.input[0]:
            buffer = self.input[0].pop()
            if type(buffer) == OVSignalBuffer:
                if (buffer):
                    samples = int(self.epoch_duration * self.fs)
                    channels = int(len(buffer) / samples)
                    epoch = np.array(buffer).reshape(channels, samples)
                    r = apply_cca(epoch, self.references)
                    command = predict(r)
                    # self.command = str(self.frequencies[command+1])
                    self.command = str(command+1)
                    self.do_feedback = True
                    print('Frequency detected %s Hz' %(self.frequencies[command+1]))  
                    



    
    def uninitialize(self):
        pass


box = SSVEPpredictor()