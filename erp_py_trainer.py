from __future__ import print_function, division
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
# from sklearn.lda import LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import KFold, cross_val_score
import numpy as np


# Training function
def train(x, y):
    print("Training...")
    clf = make_pipeline(preprocessing.StandardScaler(), LDA(solver='lsqr', shrinkage='auto'))
    cv = KFold(n_splits=5)
    scores = cross_val_score(clf, x, y, cv=cv)
    print("End training")
    return scores
    # model = 
    # return model


class ClassifierTrainer(OVBox):
    def __init__(self):
        super(ClassifierTrainer, self).__init__()
        self.stims = []
        self.buffers_target = 0
        self.buffers_nontarget = 0
        self.do_train = False
        self.x = []
        self.y = [] 


    def initialize(self):
        pass

    
    def process(self):        
        # collect stimulations, keep only target/non_targets ones and binarize them
        
        if self.input[0]:
            chunk = self.input[0].pop()
            if type(chunk) == OVStimulationSet:
                if (chunk):
                    stim = chunk.pop()
                    if stim.identifier == OpenViBE_stimulation['OVTK_StimulationId_Target'] or stim.identifier == OpenViBE_stimulation['OVTK_StimulationId_NonTarget']: 
                        self.stims.append(stim.identifier - OpenViBE_stimulation['OVTK_StimulationId_Target'])
                    
                    if stim.identifier == OpenViBE_stimulation['OVTK_StimulationId_ExperimentStop']:
                        self.do_train = True
                    

        # stack Target feature vectors
        if self.input[1]:
            x1 = self.input[1].pop()
            if type(x1) == OVStreamedMatrixBuffer:
                if (x1):
                    self.buffers_target += 1
                    self.x.append(x1)

        # stack Non Target features vectors
        if self.input[2]:
            x2 = self.input[2].pop()
            if type(x2) == OVStreamedMatrixBuffer:
                if (x2):
                    self.buffers_nontarget += 1
                    self.x.append(x2)

        
        # Train after Experiments end
        if self.do_train:
            self.y = np.array(self.stims)
            self.x = np.array(self.x)  
            scores = train(self.x, self.y) 
            print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
            stimSet = OVStimulationSet(self.getCurrentTime(),
                                   self.getCurrentTime())    
            stimSet.append(OVStimulation(OpenViBE_stimulation['OVTK_StimulationId_ExperimentStop'], 
                            self.getCurrentTime(), 
                            self.getCurrentTime()))
            self.output[0].append(stimSet)
            self.do_train = False



    def uninitialize(self):
        pass


box = ClassifierTrainer()