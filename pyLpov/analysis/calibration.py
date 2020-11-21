from pyLpov.paradigms import base
from pyLpov.io.dataset import DataSet
from pyLpov.machine_learning.evaluate import evaluate_pipeline
import pandas as pd
import numpy as np
import json
import pickle


class Calibration(object):
    
    def __init__(self, datapath=None, experiment=None, pipeline=None, 
                 outdir=None, paradigm=None, model=None, fitted=False
                ):
        self.datapath = datapath
        self.experiment = experiment
        self.pipeline = pipeline
        self.outdir = outdir
        if paradigm:
            self.paradigm = paradigm            
        else:
            self.paradigm = self.parse_paradigm(self.experiment)
        self.model = None
        self.fitted = False                
    
    @staticmethod
    def parse_paradigm(exp):
        with open(exp, "r") as read_file:
            configs = json.load(read_file)

        paradigm_type = base.PARADIGM_TYPE[configs['paradigmType']]
        md = paradigm_type.lower()
        mod = __import__('pyLpov.paradigms.'+md, fromlist=paradigm_type)
        if  paradigm_type == 'HYBRID':
            paradigm = getattr(mod, paradigm_type)().from_json(configs)
        else:
            paradigm = getattr(mod, paradigm_type).from_json(configs)        
        return paradigm

    
    def run_analysis(self, dataset=None):
        # cnt, dataset, session_interval = DataSet.convert_raw(self.datapath, self.paradigm)
        # dataset.get_epochs(cnt, session_interval, self.pipeline['fitler'])
        if not dataset:
            cnt, dataset = DataSet.convert_raw(self.datapath, self.paradigm)
            dataset.get_epochs(cnt, self.pipeline['filter'])
            self.raw_converted = True

        # FIXME
        # cross-validate
        if self.pipeline['pipeline'].steps[-1][0] == 'gridsearchcv':
            # TODO
            dataset.epochs = dataset.epochs.transpose((2,1,0))
            self.model, results = evaluate_pipeline(dataset.epochs, dataset.y, self.pipeline['pipeline'])  
        else:
            self.model = self.pipeline['pipeline']         
        # fit
        print("fitting data of shapes: ", dataset.epochs.shape, dataset.y.shape)
        self.model.fit(dataset.epochs, dataset.y)        
        self.fitted = True


    def save(self):
        # save fitted model to outdir             
        model_name = '_'.join([self.paradigm.paradigmType, self.pipeline['pipeline'].steps[-1][0],'py.clf'])
        filename = '\\'.join([self.outdir, model_name])
        if self.fitted:
            # TODO
            pickle.dump(self.model, open(filename, 'wb'))
            print('model saved at: ', filename)
        else:
            print('Model not fitted yet')

        

    