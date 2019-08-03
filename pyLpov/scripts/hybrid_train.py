from __future__ import print_function, division
from sklearn.metrics import confusion_matrix
from pyLpov.machine_learning import cca, base, evaluate
from pyLpov.proc import processing
from pyLpov.utils import utils
import numpy as np
# utils
import pickle
import os


OVTK_StimulationLabel_Base = 0x00008100

class HybridClassifierTrainer(OVBox):

    def __init__(self):
        super(HybridClassifierTrainer, self).__init__()
        self.fs = 512
        self.channels = 0
        # self.signal = []
        self.signal =np.array([])
        self.tmp_list = []
        self.pipelines = None
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
        self.pipelines = utils.parse_config_file(self.setting["Config File"])
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
            erp_signal = processing.eeg_filter(erp_signal, self.fs, self.erp_lowPass, self.erp_highPass, self.erp_filterOrder)            
            self.signal[self.erp_begin:self.erp_end,:] = erp_signal
       
            erp_epochs = [] 
            for i in range(self.erp_stims_time.shape[0]):
                erp_epochs.append(processing.eeg_epoch(self.signal, np.array([0, self.erp_epochDuration],dtype=int), self.erp_stims_time[i,:]))
            erp_epochs = np.array(erp_epochs).transpose((1,2,3,0))
            # self.erp_x = eeg_feature(erp_epochs, self.erp_downSample, self.erp_movingAverage)
            ## for use with BLDA
            if self.pipelines['ERP_pipeline'].steps[-1][0] == 'blda':
                samples, channels, epochs, trials = erp_epochs.shape
                self.erp_x = erp_epochs.reshape((samples, channels, epochs*trials), order='F')
                self.erp_model =  self.pipelines['ERP_pipeline']
            else:
                # TODO 
                # feature extraction / cross_validate pipeline
                # # self.erp_x = eeg_feature(erp_epochs, self.erp_downSample, self.erp_movingAverage)  
                pass        
            ##
            self.erp_y = np.array(self.erp_stims, dtype=np.float32)
            self.erp_y[self.erp_y==1] = -1
            self.erp_y[self.erp_y==0] = 1            
            
            del erp_signal
            del erp_epochs
            # SSVEP
            print('SSVEP analysis')
            ssvep_signal = self.signal[self.ssvep_begin:self.ssvep_end,:]
            ssvep_signal = processing.eeg_filter(ssvep_signal, self.fs, self.ssvep_lowPass, self.ssvep_highPass, self.ssvep_filterOrder)            
            self.signal[self.ssvep_begin:self.ssvep_end,:] = ssvep_signal
            ssvep_epochs = processing.eeg_epoch(self.signal, np.array([0, self.ssvep_samples-1],dtype=int), self.ssvep_stims_time)
            ssvep_predictions = []
            self.ssvep_x = ssvep_epochs
            
            if self.ssvep_mode == 'sync':                
                sync_trials = np.where(self.ssvep_y != 1)
                ssvep_sync_epochs = ssvep_epochs[:,:,sync_trials].squeeze()                
                for i in range(ssvep_sync_epochs.shape[2]):
                    ssvep_predictions.append(cca.predict(cca.apply_cca(ssvep_sync_epochs[0:self.ssvep_samples,:,i].transpose((1,0)), self.ssvep_references)) + 1)
                ssvep_targets = np.array(self.ssvep_y[sync_trials]-1)
                
            elif self.ssvep_mode == 'async':           
                # self.ssvep_model, ssvep_results = train_ssvep(self.ssvep_x, self.ssvep_y)
                if self.pipelines['SSVEP_pipeline'].steps[0][0] == 'mlr':
                    self.ssvep_x = self.ssvep_x.transpose((2,1,0))
                self.ssvep_model, ssvep_results = evaluate.evaluate_pipeline(self.ssvep_x, self.ssvep_y, self.pipelines['SSVEP_pipeline'])
          
            print("Train Accuracy: %0.2f (+/- %0.2f)" % (ssvep_results['train_score'].mean(), ssvep_results['train_score'].std() * 2))
            print("Val Accuracy: %0.2f (+/- %0.2f)" % (ssvep_results['test_score'].mean(), ssvep_results['test_score'].std() * 2))
            
            del ssvep_signal
            del ssvep_epochs 
            self.do_train = False
            self.do_save = True

        if self.do_save:
            
            self.erp_model.fit(self.erp_x, self.erp_y)
            # erp_model = self.pipelines['ERP_pipeline'].fit(self.erp_x, self.erp_y)
            # erp_model = train(self.erp_x, self.erp_y)           
            # erp_model.fit(self.erp_x, self.erp_y)
            base.save_model(self.erp_model_path, self.erp_model)
            #
            self.ssvep_model.fit(self.ssvep_x, self.ssvep_y)
            cm = confusion_matrix(self.ssvep_y, self.ssvep_model.predict(self.ssvep_x))
            print('SSVEP Confusion matrix:\n ', cm)
            base.save_model(self.ssvep_model_path, self.ssvep_model)
            #
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