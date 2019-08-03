from __future__ import print_function, division
from scipy.linalg import eig
from scipy import sqrt
import scipy.signal as sig
import numpy as np
import pickle
import os



class Inspector(OVBox):

    def __init__(self):
        super(Inspector, self).__init__()
        self.signal =np.array([])
        self.chunk = 0
        self.channels = 0
        self.nChunks = 0
        self.ends = 0
        self.do_train =  False
        self.file = None

    def initialize(self):
        pass

    def process(self):
        
        # stream signal
        for chunkIndex in range( len(self.input[0]) ):
           
            # signal_chunk = self.input[0][chunkIndex]
    
            # if type(signal_chunk) == OVSignalHeader:
            if(type(self.input[0][chunkIndex]) == OVSignalHeader):
                signal_chunk = self.input[0].pop()
                self.channels, self.chunk = signal_chunk.dimensionSizes
                
                # self.file = open('testfile3.txt','w')
                # print(signal_chunk.__dict__)

            # elif type(signal_chunk) == OVSignalBuffer:                
            elif(type(self.input[0][chunkIndex]) == OVSignalBuffer):
                # self.signal.append(signal_chunk)
                # signal_chunk = self.input[0].pop()
                signal_chunk = np.array(self.input[0].pop()).reshape((self.channels, self.chunk)) 
                # print(signal_chunk.__dict__)
                self.signal = np.hstack((self.signal, signal_chunk)) if self.signal.size else signal_chunk
                #print(signal_chunk)
                #sig = np.array(self.input[0].pop()).reshape((self.channels, self.chunk)).T
                #s = str(sig) 
                #self.file.write(s)
                # print("chunk %d shape %s sig: %f" %(self.nChunks, sig.shape, sig[0,0]))
                # self.signal.append(self.input[0].pop())
                self.nChunks += 1


        # collect Stimulations markers and times for each paradigm
        if self.input[1]:
            chunk = self.input[1].pop()
            if type(chunk) == OVStimulationSet:
                for stimIdx in range(len(chunk)):
                    if chunk:
                        stim = chunk.pop()

                        if stim.identifier == OpenViBE_stimulation['OVTK_StimulationId_ExperimentStop']:
                            self.ends += 1                                           
                        
                        if self.ends == 2:
                            self.do_train = True    

        if self.do_train:
            # self.signal = np.array(self.signal).reshape(self.channels, self.chunk*self.nChunks).T   
            pickle.dump(self.signal, open('sig_inspec_stack', 'wb'))
            self.do_save = False
            stimSet = OVStimulationSet(0.,0.)    
            stimSet.append(OVStimulation(OpenViBE_stimulation['OVTK_StimulationId_ExperimentStop'], 0.,0.)) 
            self.output[0].append(stimSet)

    def uninitialize(self):
        pass


box = Inspector()