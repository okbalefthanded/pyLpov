from __future__ import print_function, division
from aawedha.utils.utils import log
from sklearn.metrics import confusion_matrix
# from pyLpov.proc import processing
from pyLpov.proc.processing import eeg_filter, eeg_epoch
# from pyLpov.machine_learning.cca import CCA
from baseline.ssvep.cca import CCA
from pyLpov.io.models import load_model, predict_openvino_model
from pyLpov.utils.experiment import serialize_experiment, update_experiment_info
from sklearn.metrics import accuracy_score, roc_auc_score
# from pyLpov.utils.utils import is_keras_model
# from tensorflow.keras.models import load_model
import numpy as np
# import pandas as pd
# import logging
# import pickle
import socket
import random
import torch
import os


# np.set_printoptions(precision=4)
OVTK_StimulationLabel_Base = 0x00008100

class SSVEPpredictor(OVBox):

    def __init__(self):
        super(SSVEPpredictor, self).__init__()
        self.experiment_mode = 'copy'
        self.signal = np.array([])
        self.channels = 0
        self.tmp_list = []
        self.n_trials = 0
        self.ssvep_model = None
        self.idle_model = None
        self.model_path = None
        self.deep_model = False
        self.model_file_type = ''
        # self.frequencies = ['idle', 16, 16.75, 17.25, 18]
        # self.frequencies = ['idle', 9, 9.75, 10.5, 11.25]
        # self.frequencies = ['idle', 9.25, 9.75, 10.25, 10.75]
        # self.frequencies = ['idle', 8, 9, 10, 11]
        self.frequencies = ['idle', 9, 8, 10, 11]
        # self.frequencies = ['idle', 8, 9, 10]
        # self.frequencies = ['idle', 10, 9, 8]
        # self.frequencies = ['idle', 11, 10, 9, 8]
        # self.frequencies = ['idle', 9.5, 9, 8.5, 8]
        # self.frequencies = ['idle', 8.57, 6.67, 12, 5.45]
        # self.frequencies = ['idle', 8, 8.75, 9.5, 10.25]
        # self.frequencies = ['idle', 8.25, 9, 9.75, 10.5]
        self.tr_dur = []
        #
        self.ssvep_stims = []
        self.ssvep_stims_time = []
        self.ssvep_x = []
        self.ssvep_y = []
        self.low_pass = 4
        self.high_pass = 60
        self.filter_order = 2
        self.epoch_duration = 1.0
        self.ssvep_begin = 0
        self.ssvep_end = 0
        self.ssvep_correct = 0
        self.ssvep_target = []
        self.ssvep_pred = []
        #
        self.async_trial = False
        self.mode = 'sync'      
        self.harmonics = 0
        self.fs = 512
        self.samples = 0
        self.references = []
        self.async_dur = 0.6 # .6 # 600 ms
        self.async_slide = .1 # 100 ms 
        self.idle_count = 0
        self.idle_preds = []
        self.idle_cmd = []
        self.class_offset = 1
        #
        self.ends = 0
        self.nChunks = 0
        self.chunk = 0
        #
        self.command = None
        self.hostname = '127.0.0.1'
        self.feedback_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.feedback_port = 12345   
        self.do_feedback = False 
        self.trial_ended = False
        # 
        self.dur  = 0
        self.file_name = ''
        self.is_replay = False
        #
        # self.log = log("async_log_stim_dur.log", logger_name="async_online_log")

    def initialize(self):
        #
        if 'File Name' in self.setting:
            self.file_name = f"{self.setting['File Name']}.json"
        else:
            self.is_replay = True
        self.experiment_mode = self.setting["Experiment Mode"]
        self.fs = int(self.setting['Sample Rate'])
        # self.epoch_duration = np.floor(float(self.setting['Epoch Duration (in sec)']))
        # self.epoch_duration = float(self.setting['Epoch Duration (in sec)'])
        self.epoch_duration = np.array(self.setting['Epoch Duration (in sec)'].split(','), dtype=np.float)
        self.low_pass = int(self.setting["Low Pass"])
        self.high_pass = int(self.setting["High Pass"])
        self.filter_order = int(self.setting["Filter Order"])
        self.harmonics = int(self.setting['Harmonics'])
        self.mode = self.setting["Mode"]
        self.model_path = self.setting["Classifier"]
        if len(self.epoch_duration) >  1:
            dur = np.diff(self.epoch_duration)
            self.samples = int(dur * self.fs)    
            self.dur = (self.epoch_duration * self.fs).astype(int)        
        else:
            dur = self.epoch_duration
            self.samples = int(self.epoch_duration * self.fs)
            self.dur = np.array([0, self.epoch_duration[0]*self.fs], dtype=int)

        t = np.arange(0.0, float(self.samples)) / self.fs
        if self.mode == 'sync':
            frequencies = self.frequencies[1:]
            self.frequencies = self.frequencies[1:]
            # self.frequencies = np.arange(9.25, 15., 0.5).tolist()
            # frequencies = self.frequencies
            # generate reference signals
            x = [ [np.cos(2*np.pi*f*t*i),np.sin(2*np.pi*f*t*i)] for f in frequencies for i in range(1, self.harmonics+1)]
            self.references = np.array(x).reshape(len(frequencies), 2*self.harmonics, self.samples)
            # self.ssvep_model = CCA(self.harmonics, frequencies, self.references, int(self.epoch_duration))
            self.ssvep_model = CCA(self.harmonics, frequencies, phase=None, references=self.references, length=int(dur))
            
        # elif self.mode == 'async' or self.mode == 'sync_train':
        else:
            self.ssvep_model, self.deep_model, self.model_file_type = load_model(self.model_path)
            # check idle model (binary classification idle vs other_freqs)
            if self.mode == "async_dynamic" or self.mode == "async_static":
                ending = 4
                if self.model_path.count("sync") == 2:
                    ending = 9                
                if self.mode == "async_dynamic":
                    idlepath = f"{self.model_path[:-ending]}_idle.pth"
                elif self.mode == "async_static":
                    idlepath = f"{self.model_path[:-ending]}_idles.pth"                
           
                if os.path.exists(idlepath):
                    self.class_offset = 2                
                    self.idle_model = torch.load(idlepath, map_location=torch.device("cpu"))
                    self.idle_model.set_device("cpu")
                    self.idle_model.eval()
            '''
            if is_keras_model(self.model_path):
                # Keras model
                self.ssvep_model = load_model(self.model_path)
                self.keras_model = True
            else:
                # sklearn model
                self.ssvep_model = pickle.load(open(self.model_path, 'rb'),  encoding='latin1') #py3
            '''
            # self.ssvep_model = None
            # self.ssvep_model = pickle.load(open(self.ssvep_model_path, 'rb')) #py2
            # generate reference by itcca method
            self.references = self.ssvep_model
        # create json file
        info = {'title': "SSVEP_LARESI",
                'stimulation' : int(self.epoch_duration[1]*1000),
                'break_duration' : int(self.epoch_duration[0]*1000),
                'repetition': 0,
                'stimuli' : len(self.frequencies),
                'phrase' : [],
                'stim_type' : 'Sinusoidal',
                'frequencies' : [str(f) for f in self.frequencies],
                'control' : self.mode 
                }
        if not self.is_replay:
            serialize_experiment(self.file_name, info)

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
        self.ssvep_stims = np.array(self.ssvep_stims)                          
        # self.ssvep_end = int(np.ceil(stim.date * self.fs))
        # self.ssvep_end = np.ceil(stim.date * self.fs).astype(int) 
        # self.ssvep_y = np.array(self.ssvep_y) 
        # mrk = np.array(self.ssvep_stims_time).astype(int) - self.ssvep_begin
        mrk = np.array(self.ssvep_stims_time) - self.ssvep_begin
        
        # ssvep_signal = processing.eeg_filter(self.signal[:, self.ssvep_begin:self.ssvep_end].T, self.fs, self.low_pass, self.high_pass, self.filter_order)                            
        ssvep_signal = eeg_filter(self.signal[:, self.ssvep_begin:self.ssvep_end].T, self.fs, self.low_pass, self.high_pass, self.filter_order)                            
        # ssvep_signal = bandpass(self.signal[:, self.ssvep_begin:self.ssvep_end].T, [self.low_pass, self.high_pass], self.fs, self.filter_order)                          
        # print(f"dur : {dur}")
        # ssvep_epochs = processing.eeg_epoch(ssvep_signal, dur, mrk, self.fs)
        # print(f"[Filter and epoch] : mrk {mrk} signal {ssvep_signal.shape} dur {self.dur}")
        # ssvep_epochs = eeg_epoch(ssvep_signal, self.dur, mrk, self.fs) 
        # ssvep_epochs = processing.eeg_epoch(ssvep_signal, np.array([0, self.samples],dtype=int), mrk, self.fs)
        # ssvep_epochs = processing.eeg_epoch(ssvep_signal, np.array([int(0.1*self.fs), self.samples],dtype=int), mrk, self.fs)
        # self.ssvep_x = ssvep_epochs.squeeze()
        # self.ssvep_x = eeg_epoch(ssvep_signal, self.dur, mrk, self.fs).squeeze()
        '''
        if (mrk + self.dur.max()) > ssvep_signal.shape[0]:
            dur = np.array([0. ,0.5 * self.fs], dtype=int)
        else:
            dur = self.dur
        '''
        dur = self.dur
        self.ssvep_x = eeg_epoch(ssvep_signal, dur, mrk, self.fs).squeeze() #.astype(np.float16)
        # self.ssvep_x = eeg_epoch(ssvep_signal, self.dur, mrk, self.fs).squeeze().astype(np.float16)
        # self.ssvep_x  = eeg_epoch(ssvep_signal, self.dur, mrk, self.fs, baseline_correction=True).squeeze()
        del ssvep_signal
        # del ssvep_epochs
        del mrk        
    
    def predict(self):
        '''
        '''
        if self.mode == 'sync':                
            # sync_trials = np.where(self.ssvep_y != 1)
            # ssvep_sync_epochs = ssvep_epochs[:,:,sync_trials].squeeze()
            # ssvep_sync_epochs = self.ssvep_x
            # print(f"sync epochs shape: {ssvep_sync_epochs.shape}, samples {self.samples}")
            ssvep_predictions = np.array(self.ssvep_model.predict(self.ssvep_x[0:self.samples, :].transpose((1,0))) + 1)
            # ssvep_predictions = self.ssvep_model.predict(ssvep_sync_epochs[51:self.samples,:].transpose((1,0))) + 1                                  
            # ssvep_predictions = np.array(ssvep_predictions)                      
        # elif self.mode == 'async' or self.mode == 'sync_train' :
        # adding more modes: async | sync_train | async_dynamic
        else:
            if self.deep_model:   
                if self.model_file_type == 'pth':
                    ssvep_predictions = self.ssvep_model.predict(self.ssvep_x[..., None].transpose((2, 1, 0)), normalize=True).argmax() + self.class_offset #1                     
                else:                
                    ssvep_predictions = self.ssvep_model.predict(self.ssvep_x[..., None].transpose((2, 1, 0))).argmax() + self.class_offset # 1
            else:
                ssvep_predictions = self.ssvep_model.predict(self.ssvep_x[..., None]) + 1
            
            # ssvep_predictions = np.array([1])

        self.command = str(ssvep_predictions.item())

    def predict_idle(self, ssvep_x):
        return self.idle_model.predict(ssvep_x[..., None].transpose((2, 1, 0)), normalize=True)

    def print_if_target(self):
        '''
        '''
        if self.experiment_mode == 'Copy':
            print('[SSVEP] preds:', self.command, ' target:', self.ssvep_y)
            # self.log.debug(f"[SSVEP] preds:, {self.command},  target:, {self.ssvep_y}, idle score: {self.idle_preds}")
            if int(self.command) == self.ssvep_y:
                self.ssvep_correct += 1
            self.ssvep_target.append(self.ssvep_y[-1])

    def init_data(self):
        '''
        '''
        self.ssvep_x = []
        self.ssvep_y = []
        self.tmp_list = []
        self.ssvep_stims = []
        self.ssvep_stims_time = []   
        self.idle_preds = [] 
        self.idle_count = 0 
        self.do_feedback = False                
        self.n_trials += 1

    def print_results(self):
        '''
        '''        
        print('Trial N :', self.n_trials, ' / SSVEP Target : ', self.ssvep_target)
        print('Trial N :', self.n_trials, ' / SSVEP Pred : ', self.ssvep_pred)
        print('Trial N :', self.n_trials, ' / SSVEP Accuracy : ', (self.ssvep_correct / self.n_trials) * 100)

    def experiment_end(self):
        '''
        '''
        print('EXPERIMENT ENDS')
        # update experiment json file
        if not self.is_replay:
            update_experiment_info(self.file_name, "repetition", self.n_trials // len(self.frequencies))

        if self.experiment_mode == 'Copy':
            self.ssvep_target = np.array(self.ssvep_target)
            if self.mode == "async_dynamic" or self.mode == "async_static":
                self.idle_cmd = np.array(self.idle_cmd)
                y_idle = np.zeros_like(self.ssvep_target)
                y_idle[(self.ssvep_target - 1) != 0] = 1.
                idle_p = np.zeros_like(self.idle_cmd)
                idle_p[self.idle_cmd > .5] = 1. 
                print(f"Idle accuracy : {accuracy_score(y_idle, idle_p)*100} AUC: {roc_auc_score(y_idle, idle_p)}")
                print(f"Idle cm: {confusion_matrix(y_idle, idle_p)}")             
                      
            print(' SSVEP Accuracy : ', (self.ssvep_correct / self.n_trials) * 100)
            cm = confusion_matrix(self.ssvep_target, np.array(self.ssvep_pred))
            print('Confusion matrix: ', cm)
        
        self.switch = False
        del self.signal
        del self.ssvep_x
        del self.ssvep_model
        jitter = np.diff(self.tr_dur)
        print('Trial durations delay: ',  jitter, jitter.min(), jitter.max(), jitter.mean())
        stimSet = OVStimulationSet(0.,0.)    
        stimSet.append(OVStimulation(OpenViBE_stimulation['OVTK_StimulationId_ExperimentStop'], 0., 0.)) 
        self.output[0].append(stimSet)     
       
    def async_limit(self):
        """Test if Trial length reached the duration of 500 (or any async_dur fixed) ms (used for async mode)
        """
        # dur = self.signal.shape[1] - self.ssvep_begin
        if self.ssvep_stims_time:
            dur = self.signal.shape[1] - self.ssvep_stims_time[-1]  - (self.idle_count * int(self.async_slide*self.fs) )
            # print(dur, self.signal.shape[1], self.idle_count) # = 307 = 0.6 ms (512 fs)
            # self.log.debug(f"dur {dur} signal shape {self.signal.shape[1]} idle count {self.idle_count}")
            self.async_trial = True
            # return dur >= ( (self.async_dur * self.fs * (self.idle_count+1)) )
            return dur >= (self.async_dur * self.fs)        
    
    def process(self):
        
        # stream signal
        if self.input[0]:            
            self.stream_signal()   

        if self.mode == 'async_dynamic':
            if self.async_limit() and self.ssvep_stims_time:
                self.idle_count += 1
                # filter, epoch, predict                               
                idle_tmp_end = self.ssvep_stims_time[-1] + (self.idle_count * int(self.async_dur*self.fs))
                ssvep_signal = eeg_filter(self.signal[:,self.ssvep_begin:idle_tmp_end].T, self.fs, self.low_pass, self.high_pass, self.filter_order)
                mrk = np.array(self.ssvep_stims_time) - self.ssvep_begin + int(self.fs* self.async_slide*(self.idle_count-1))
                dr =  np.array([0, self.async_dur*self.fs], dtype=int)            
                
                ssvep_x = eeg_epoch(ssvep_signal, dr, mrk, self.fs).squeeze()
                self.idle_preds.append(self.predict_idle(ssvep_x)[0])      
                
                if len(self.idle_preds) == 3: # 2 , 3 number of consecutive idle trials
                    idle_score = max(self.idle_preds)
                    # print(f"IDLESCORE : {idle_score} {self.idle_preds}")
                    if idle_score < 0.5:
                        self.do_feedback = True
                        self.feedback_socket.sendto("1".encode(), (self.hostname, self.feedback_port))
                        # print("feedback sent")                        
                        self.ssvep_pred.append(1)
                        self.command = "1"
                '''
                self.do_feedback = True
                self.feedback_socket.sendto("1".encode(), (self.hostname, self.feedback_port))                                               
                self.ssvep_pred.append(1)
                self.command = "1"
                self.idle_preds = [[1]]
                '''
        # collect Stimulations markers and times for each paradigm
        if self.input[1]:
            chunk = self.input[1].pop()
            if type(chunk) == OVStimulationSet:
                for stimIdx in range(len(chunk)):
                    if chunk:
                        stim = chunk.pop()               
                        # print('Received Marker: ', stim.identifier, 'stamped at', stim.date, 's')                        
                        # SSVEP session                        
                        if(stim.identifier == OpenViBE_stimulation['OVTK_StimulationId_TrialStart']):
                            self.do_feedback = False                       
                            print('[SSVEP trial start]', stim.date)
                            if(len(self.ssvep_stims_time) == 0):
                                self.tr_dur.append(stim.date)
                                # self.ssvep_begin = int(np.floor(stim.date * self.fs))
                                self.ssvep_begin = int(stim.date * self.fs)

                        if (stim.identifier == OpenViBE_stimulation['OVTK_StimulationId_VisualSteadyStateStimulationStart']):
                            print('[SSVEP Visual_stim_start]', stim.date)
                            # self.ssvep_stims_time.append(np.floor(stim.date*self.fs).astype(int))
                            self.ssvep_stims_time.append(int(stim.date*self.fs))

                        if (stim.identifier >= OVTK_StimulationLabel_Base) and (stim.identifier <= OVTK_StimulationLabel_Base+len(self.frequencies)):
                            self.ssvep_y.append(stim.identifier - OVTK_StimulationLabel_Base)
                            print('[SSVEP stim]', stim.date, self.ssvep_y[-1])
                            # self.ssvep_stims_time.append(np.floor(stim.date*self.fs).astype(int))                     
                        
                        if(stim.identifier == OpenViBE_stimulation['OVTK_StimulationId_TrialStop']):                            
                            # print(f"[idle count] {self.idle_count}")
                            self.ssvep_y = np.array(self.ssvep_y) 
                            print('[SSVEP trial stop]', stim.date)
                            print('[Stim duration:]', stim.date - self.ssvep_stims_time[0] / self.fs)
                            
                            # self.ssvep_end = np.ceil(stim.date * self.fs).astype(int)
                            if self.mode == "async_dynamic":
                                self.idle_cmd.append(max(self.idle_preds)[0])
                            # async | async_static | sync predict
                            if not self.do_feedback:
                                self.ssvep_end = int(stim.date * self.fs)                             
                                self.filter_and_epoch(stim)
                                self.predict()
                                if self.mode == "async_static":
                                    idle_p = self.predict_idle(self.ssvep_x)                             
                                    if idle_p < .5:
                                        self.command = "1"                                                             
                                    self.idle_cmd.append(idle_p[0])                        
                                # commands = ['1', '2', '3', '4']
                                # self.command = random.choice(commands)
                                # print(f"idle preds: {max(self.idle_preds)}")                                
                                print('[SSVEP] Command to send is: ', self.command)
                                self.feedback_socket.sendto(self.command.encode(), (self.hostname, self.feedback_port))                                                           
                                self.ssvep_pred.append(int(self.command))
                            
                            self.print_if_target()  
                            self.init_data()
                            # self.print_results()                                                      

                        # Ending Experiment 
                        if stim.identifier == OpenViBE_stimulation['OVTK_StimulationId_ExperimentStop']:
                            self.experiment_end()

            del chunk
        return        

    def unintialize(self):
        pass


box = SSVEPpredictor()
