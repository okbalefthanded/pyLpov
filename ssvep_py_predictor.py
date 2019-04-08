from __future__ import print_function, division
from sklearn.cross_decomposition import CCA
import numpy as np
import pickle



class SSVEPpredictor(OVBox):

    def __init__(self):
        super(SSVEPpredictor, self).__init__()
        self.model = None
        self.predictions = []
        self.events = []
        self.trials_count = 0
        self.frequencies = ['idle', 6.66, 7.5, 8.57, 10]
        self.num_harmonics = 0
        self.epoch_duration = 0
        self.fs = 512
        self.references = []      


    def initialize(self):
        self.epoch_duration = float(self.setting['Epoch_duration'])
        self.num_harmonics = int(self.setting['Harmonics'])
        self.fs = float(self.setting['Sample_rate'])
        samples = self.epoch_duration * self.fs
        t = np.arange(0.0, samples) / self.fs
        if self.frequencies[0] == 'idle':
            frequencies = self.frequencies[1:]
        # generate reference signals
        x = [[np.cos(2*np.pi*f*t),np.sin(2*np.pi*f*t)] for f in frequencies]
        self.references = np.array(x).reshape(self.num_harmonics * len(frequencies), int(samples))  
        

    
    def process(self):
        # 
        if self.input[1]:
            chunk = self.input[1].pop()
            if type(chunk) == OVStimulationSet:
                for stimIdx in range(len(chunk)):
                    if chunk:
                        stim = chunk.pop()
                        print('Received Marker: ', stim.identifier, 'stamped at', stim.date, 's')

        if self.input[0]:
            buffer = self.input[0].pop()
            if type(buffer) == OVSignalBuffer:
                if (buffer):
                    epoch = np.array(buffer)
                    # model = CCA(n_component=self.num_harmonics)
                    model = CCA(n_components=self.num_harmonics).fit(epoch, self.references).transform(epoch)
                    



    
    def uninitialize(self):
        pass


box = SSVEPpredictor()