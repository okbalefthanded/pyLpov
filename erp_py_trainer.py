from __future__ import print_function, division
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import KFold, cross_validate
import numpy as np
import pickle
import os


# Training function
def train(x, y):
    print("Training...")
    clf = make_pipeline(preprocessing.StandardScaler(), LDA(solver='lsqr', shrinkage='auto'))
    cv = KFold(n_splits=5)
    #scores = cross_val_score(clf, x, y, cv=cv)
    cv_results = cross_validate(clf, x, y, cv=cv, 
                                scoring=('accuracy', 'roc_auc'),
                                return_train_score=True)
    print("End training")
    return clf, cv_results


class ClassifierTrainer(OVBox):
    def __init__(self):
        super(ClassifierTrainer, self).__init__()
        self.stims = []
        self.do_train = False
        self.do_save = False
        self.x = []
        self.y = [] 


    def initialize(self):
        pass

    # Save model after training
    def save_model(self, model):
        filename = self.setting['classifier_path']
        pickle.dump(model, open(filename, 'wb'))

    
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
                    self.x.append(x1)

        # stack Non Target features vectors
        if self.input[2]:
            x2 = self.input[2].pop()
            if type(x2) == OVStreamedMatrixBuffer:
                if (x2):
                    self.x.append(x2)

        
        # Cross-validate after Experiments end
        if self.do_train:
            self.y = np.array(self.stims)
            self.x = np.array(self.x)  
            model, scores = train(self.x, self.y) 
            print("Train Accuracy: %0.2f (+/- %0.2f)" % (scores['train_accuracy'].mean(), scores['train_accuracy'].std() * 2))
            print("Val Accuracy: %0.2f (+/- %0.2f)" % (scores['test_accuracy'].mean(), scores['test_accuracy'].std() * 2))
            print("Train ROC: %0.2f (+/- %0.2f)" % (scores['train_roc_auc'].mean(), scores['train_roc_auc'].std() * 2))
            print("Val ROC: %0.2f (+/- %0.2f)" % (scores['test_roc_auc'].mean(), scores['test_roc_auc'].std() * 2))            
            self.do_train = False
            self.do_save = True       


        # retrain model with best param and save model
        if self.do_save:
            model.fit(self.x, self.y)
            self.save_model(model)
            self.do_save = False
            # output a stop stimulation to Stop scenario
            stimSet = OVStimulationSet(self.getCurrentTime(),
                                      self.getCurrentTime())    
            stimSet.append(OVStimulation(OpenViBE_stimulation['OVTK_StimulationId_ExperimentStop'], 
                                self.getCurrentTime(), 
                                self.getCurrentTime()))
            self.output[0].append(stimSet)



    def uninitialize(self):
        pass


box = ClassifierTrainer()