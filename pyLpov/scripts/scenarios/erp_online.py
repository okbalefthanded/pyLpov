from sklearn.metrics import confusion_matrix
from pyLpov.proc import processing
from pyLpov.utils import utils
# from tensorflow.keras.models import load_model
# from pyLpov.io.models import load_model, predict_openvino_model
from pyLpov.io.models import load_model
import numpy as np
import socket
import logging
import pickle
import datetime
import random
import os
import gc

OVTK_StimulationLabel_Base = 0x00008100
commands = ['1', '2', '3', '4', '5', '6', '7', '8', '9']

class ERPOnline(OVBox):

    def __init__(self):
        super(ERPOnline, self).__init__()
        self.experiment_state = 0   
        self.fs = 512
        self.channels = 0
        self.mode = None
        self.stimulation = "Single"
        # self.signal = []
        self.signal =np.array([])
        self.tmp_list = []
        self.n_trials = 0
        self.tr_dur = []
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
        self.keras_model = False
        self.model_file_type = ''
        self.erp_correct = 0
        self.erp_target = []
        self.erp_pred = []
        #
        #
        self.ends = 0
        self.nChunks = 0
        self.chunk = 0
        # self.switch = False
        # self.stream_signal = False  
        #
        self.feedback_data = 0
        self.hostname = '127.0.0.1'
        self.feedback_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.erp_feedback_port = 12345      

    def experiment_end(self):
        '''
        '''
        print('EXPERIMENT ENDS')
        print(' ERP Accuracy : ', (self.erp_correct / self.n_trials) * 100)                         
        self.switch = False
        del self.signal
        del self.erp_x
        del self.erp_model
        jitter = np.diff(self.tr_dur)
        print('Trial durations delay: ',  jitter, jitter.min(), jitter.max(), jitter.mean())
        stimSet = OVStimulationSet(0.,0.)    
        stimSet.append(OVStimulation(OpenViBE_stimulation['OVTK_StimulationId_ExperimentStop'], 0.,0.)) 
        self.output[0].append(stimSet)
    
    def print_results(self):
        '''
        '''        
        print('Trial N :', self.n_trials, ' / ERP Target : ', self.erp_target)
        print('Trial N :', self.n_trials, ' / ERP Pred : ', self.erp_pred)
        print('Trial N :', self.n_trials, ' / ERP Accuracy : ', (self.erp_correct / self.n_trials) * 100)
    
    def print_if_target(self):
        '''
        '''
        self.erp_y[self.erp_y == 1] = -1
        self.erp_y[self.erp_y == 0] = 1
        
        if self.mode == 'Copy':
            tg = np.where(self.erp_y == 1)                               
            print('[ERP Target] : ', self.erp_stims[tg[0][0]] )
            self.erp_target.append(self.erp_stims[tg[0][0]]) 
            if self.command == '#':
                print('NO ERP detection ...')
            elif int(self.command) == self.erp_stims[tg[0][0]]:                               
                print('[ERP Correct! ]')
                self.erp_correct += 1
    
    def init_data(self):
        '''
        '''
        self.erp_x = []
        self.erp_y = []
        self.tmp_list = []
        self.erp_stims = []
        self.erp_stims_time = []                     
        self.n_trials += 1                         
            
    def stream_signal(self):
        '''
        '''
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
    
    def filter_and_epoch(self, stim):
        '''
        '''
        # print('[current Time:]', datetime.datetime.now())
        self.erp_stims_time = self.tmp_list 
        self.erp_stims = np.array(self.erp_stims)                          
        self.erp_end = int(np.floor(stim.date * self.fs)) 
        self.erp_y = np.array(self.erp_y) 
        mrk = np.array(self.erp_stims_time).astype(int) - self.erp_begin
        step = 1
        if self.stimulation == 'Multi':
            # mrk = mrk[::3]
            step = 3
        elif self.stimulation == 'Dual':
            step = 2
        mrk = mrk[::step]
        erp_signal = processing.eeg_filter(self.signal[:, self.erp_begin:self.erp_end].T, self.fs, self.erp_lowPass, self.erp_highPass, self.erp_filterOrder)
        strt = int(0.1*self.fs)
        # np.array([strt, self.erp_epochDuration],dtype=int)
        ep = np.ceil(np.array([0.1, 0.5])*self.fs).astype(int) # FIXME
        erp_epochs = processing.eeg_epoch(erp_signal,ep , mrk, self.fs)
        self.erp_x = erp_epochs
        del erp_signal
        del erp_epochs
        del mrk

    def predict(self, commands):
        '''
        '''
        predictions = []
        nbr = 1
        if self.stimulation == 'Single':
            if self.keras_model or self.model_file_type == 'xml': 
                if self.model_file_type == 'h5':               
                    predictions = self.erp_model.predict(self.erp_x.transpose((2,1,0)))
                '''
                elif self.model_file_type == 'xml':
                    predictions = predict_openvino_model(self.erp_model, self.erp_x.transpose((2,1,0)))
                '''
                predictions[predictions > .5] = 1.
            else:
                # print("[ERP epoch shape] ", self.erp_x.shape)
                predictions = self.erp_model.predict(self.erp_x)
            self.command, idx = utils.select_target(predictions, self.erp_stims, commands)
        elif self.stimulation == 'Dual' or self.stimulation == 'Multi':
            events = np.array(self.erp_stims)
            if self.stimulation == 'Dual':
                nbr = 2
            elif self.stimulation == 'Multi':
                nbr = 3
            events = events.reshape((len(events)//nbr, nbr))
            events = np.flip(events, axis=1)
            for model in self.erp_model:
                predictions.append(model.predict(self.erp_x))
            self.command, scores = utils.select_target_multistim(np.array(predictions).T, events)
            if self.command == '0':
                self.command = '#' # there is no 0 command it's a padding with command 5
            print(scores)
            del events
            
        del predictions
        
    
    def initialize(self):
        self.fs = int(self.setting["Sample Rate"])
        #
        self.erp_lowPass = int(self.setting["ERP Low Pass"])
        self.erp_highPass = int(self.setting["ERP High Pass"])
        self.erp_filterOrder = int(self.setting["ERP Filter Order"])
        self.erp_downSample = int(self.setting["Downsample Factor"])
        self.erp_epochDuration = np.ceil(float(self.setting["ERP Epoch Duration (in sec)"]) * self.fs).astype(int)
        self.erp_movingAverage = int(self.setting["ERP Moving Average"])
        self.erp_model_path = self.setting["Classifier"]
        self.erp_model, self.keras_model, self.model_file_type = load_model(self.erp_model_path)
        '''
        if utils.is_keras_model(self.erp_model_path):
            # Keras model
            self.erp_model = load_model(self.erp_model_path)
            self.keras_model = True
        else:
            # sklearn model
            self.erp_model = pickle.load(open(self.erp_model_path, 'rb'))
        '''
        self.stimulation = str(self.setting["Stimulation"])

    def process(self):        
        
        # stream signal
        if self.input[0]:            
            self.stream_signal()        
        
        # collect Stimulations markers and times for each paradigm
        if self.input[1]:
            chunk = self.input[1].pop()
            if type(chunk) == OVStimulationSet:
                for stimIdx in range(len(chunk)):
                    if chunk:
                        stim = chunk.pop()               
                        # print('Received Marker: ', stim.identifier, 'stamped at', stim.date, 's')                        
                        # ERP session                        
                        if(stim.identifier == OpenViBE_stimulation['OVTK_StimulationId_TrialStart']):                       
                            print('[ERP trial start]', stim.date)
                            self.tr_dur.append(stim.date)
                            if(len(self.erp_stims_time) == 0):
                                self.erp_begin = int(np.floor(stim.date * self.fs))                                
                        
                        if (stim.identifier == OpenViBE_stimulation['OVTK_StimulationId_Target'] or 
                            stim.identifier == OpenViBE_stimulation['OVTK_StimulationId_NonTarget']): 
                            self.mode = 'Copy'
                            self.erp_y.append(stim.identifier - OpenViBE_stimulation['OVTK_StimulationId_Target'])
                         
                        if (stim.identifier >= OVTK_StimulationLabel_Base) and (stim.identifier <= OpenViBE_stimulation['OVTK_StimulationId_LabelEnd']) :
                            self.erp_stims.append(stim.identifier - OVTK_StimulationLabel_Base) 
                            # print('[ERP stim]', stim.date, self.erp_stims[-1])
                            self.tmp_list.append(np.floor(stim.date*self.fs))        

                        if(stim.identifier == OpenViBE_stimulation['OVTK_StimulationId_TrialStop']):                            
                            print('[ERP trial stop]', stim.date)  

                            self.filter_and_epoch(stim)
                            self.predict(commands)
                            self.command = random.choice(commands)
                            print('[ERP] Command to send is: ', self.command)
                            
                            self.feedback_socket.sendto(self.command.encode(), (self.hostname, self.erp_feedback_port))                                                 
                                                      
                            self.erp_pred.append(self.command)
                            self.print_if_target()  
                            self.init_data()
                            self.print_results()                           

                        # Ending Experiment 
                        if stim.identifier == OpenViBE_stimulation['OVTK_StimulationId_ExperimentStop']:
                            self.experiment_end()

            del chunk
        return        

    def unintialize(self):
        pass

box = ERPOnline()