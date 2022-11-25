from __future__ import print_function, division
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from pyLpov.io.models import load_model, predict_openvino_model
from pyLpov.proc.processing import eeg_filter, eeg_epoch
from pyLpov.machine_learning.cca import CCA
from pyLpov.utils import utils
import numpy as np
import logging
import random
import socket
import pickle
import torch
import os


OVTK_StimulationLabel_Base = 0x00008100
commands = ['1', '2', '3', '4', '5', '6', '7', '8', '9']

class HybridOnline(OVBox):

    def __init__(self):
        super(HybridOnline, self).__init__()
        self.experiment_state = 0   
        self.fs = 512
        self.channels = 0
        self.mode = None
        self.erp_stimulation = 'Single' 
        self.signal =np.array([])
        self.tmp_list = []
        self.n_trials = 0
        self.tr_dur = []
        self.command = None
        #
        self.erp_stims = []
        self.erp_stims_time = []
        self.erp_x = []
        self.erp_y = []
        self.erp_lowPass = 1
        self.erp_highPass = 10
        self.erp_filterOrder = 2
        self.erp_model = None
        self.erp_keras_model = False
        self.erp_model_file_type = ''
        self.erp_downSample = 4
        self.erp_movingAverage = 12
        self.erp_epochDuration = 0.7
        self.erp_samples = 0
        self.erp_channels = []
        self.erp_begin = 0
        self.erp_end = 0
        self.erp_model_path = []
        self.erp_correct = 0
        self.erp_target = []
        self.erp_pred = []
        #
        self.ssvep_x = []
        self.ssvep_y = []
        self.ssvep_stims_time = []
        self.ssvep_model = None
        self.ssvep_model_path = []
        self.ssvep_keras_model = False
        self.erp_model_file_type = ''
        self.ssvep_idle_model = None
        self.ssvep_lowPass = 5
        self.ssvep_highPass = 50
        self.ssvep_filterOrder = 6   
        self.ssvep_n_harmonics = 2
        # self.ssvep_frequencies = ['idle', 8, 8.75, 9.5, 10.25]
        # self.ssvep_frequencies = ['idle', 8.25, 9, 9.75, 10.5]
        # self.ssvep_frequencies = ['idle', 8.25, 9, 9.75]
        # self.ssvep_frequencies = ['idle', 8, 8.75, 9.5]
        # self.ssvep_frequencies = ['idle', 9, 9.75, 10.5, 11.25]
        # self.ssvep_frequencies = ['idle', 8.57,6.67,12,5.54]
        # self.ssvep_frequencies = ['idle', 8.57,6.67,12,5.54]
        # self.ssvep_frequencies = ['idle', 11, 10, 9, 8] # 21-11-2022
        # self.ssvep_frequencies = ['idle', 10, 9, 8]
        # self.ssvep_frequencies = ['idle', 8, 9, 10, 11]
        # self.ssvep_frequencies = ['idle', 9, 8, 10, 11]
        # self.ssvep_frequencies = ['idle', 9, 10, 11] # 20-11-2022
        self.ssvep_frequencies = ['idle', 11, 10.25, 9.5, 8.75, 8] # 23-11-2022

        self.ssvep_epochDuration = 4.0
        self.ssvep_samples = 0
        self.ssvep_references = []   
        self.ssvep_channels = []   
        self.ssvep_mode = 'async'  
        self.ssvep_begin = 0 
        self.ssvep_correct = 0
        self.ssvep_target = []
        self.ssvep_pred = []
        self.ssvep_async_dur = 0.6 #.6 # 600 ms
        self.ssvep_async_slide = .1 # 100 ms
        self.ssvep_idle_count = 0
        self.ssvep_idle_preds = []
        self.ssvep_idle_cmd = []
        self.ssvep_class_offset = 1
        self.ssvep_feedback = False
        #
        self.ends = 0
        self.nChunks = 0
        self.chunk = 0
        self.switch = False
        # self.stream_signal = False  
        #
        self.feedback_data = 0
        self.hostname = '127.0.0.1'
        self.feedback_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.erp_feedback_port = 12345
        self.ssvep_feedback_port = 12346        

    def initialize(self):
        self.fs = int(self.setting["Sample Rate"])
        #
        # ERP init
        self.erp_stimulation = str(self.setting['ERP Stimulation'])
        self.erp_lowPass = int(self.setting["ERP Low Pass"])
        self.erp_highPass = int(self.setting["ERP High Pass"])
        self.erp_filterOrder = int(self.setting["ERP Filter Order"])
        self.erp_downSample = int(self.setting["Downsample Factor"])
        self.erp_epochDuration = np.ceil(float(self.setting["ERP Epoch Duration (in sec)"]) * self.fs).astype(int)
        self.erp_movingAverage = int(self.setting["ERP Moving Average"])
        self.erp_model_path = self.setting["ERP Classifier"]
        # self.erp_model = pickle.load(open(self.erp_model_path, 'rb'),  encoding='latin1') # py3
        # self.erp_model = pickle.load(open(self.erp_model_path, 'rb')) #py2
        self.erp_model, self.erp_keras_model, self.erp_model_file_type = load_model(self.erp_model_path)
        #
        # SSVEP init
        self.ssvep_lowPass = int(self.setting["SSVEP Low Pass"])
        self.ssvep_highPass = int(self.setting["SSVEP High Pass"])
        self.ssvep_filterOrder = int(self.setting["SSVEP Filter Order"])
        # self.ssvep_epochDuration = float(self.setting[  "SSVEP Epoch Duration (in sec)"]) 
        self.ssvep_epochDuration = np.array(self.setting["SSVEP Epoch Duration (in sec)"].split(','), dtype=np.float)
        self.ssvep_n_harmonics = int(self.setting["SSVEP Harmonics"])
        # self.ssvep_samples = int(self.ssvep_epochDuration * self.fs)
        if len(self.ssvep_epochDuration) >  1:
            dur = np.diff(self.ssvep_epochDuration)
            self.ssvep_samples = int(dur * self.fs)            
        else:
            dur = self.ssvep_epochDuration
            self.ssvep_samples = int(self.ssvep_epochDuration * self.fs)
        self.ssvep_mode = self.setting["SSVEP Mode"]
        self.ssvep_model_path = self.setting['SSVEP Classifier']
        
        t = np.arange(0.0, float(self.ssvep_samples)) / self.fs
        if self.ssvep_mode == 'sync':
            frequencies = self.ssvep_frequencies[1:]
            self.ssvep_frequencies = self.ssvep_frequencies[1:]
            # generate reference signals
            x = [ [np.cos(2*np.pi*f*t*i),np.sin(2*np.pi*f*t*i)] for f in frequencies for i in range(1, self.ssvep_n_harmonics+1)]
            self.ssvep_references = np.array(x).reshape(len(frequencies), 2*self.ssvep_n_harmonics, self.ssvep_samples)
            # self.ssvep_model = CCA(self.ssvep_n_harmonics, frequencies, self.ssvep_references, int(self.ssvep_epochDuration))
            self.ssvep_model = CCA(self.ssvep_n_harmonics, frequencies, self.ssvep_references, int(dur))
        # elif self.ssvep_mode == 'async' or self.ssvep_mode == 'sync_train':
        else:
            self.ssvep_model, self.ssvep_keras_model, self.ssvep_model_file_type = load_model(self.ssvep_model_path)
            if self.ssvep_mode == "async_dynamic" or self.ssvep_mode == "async_static":
                ending = 4
                if self.ssvep_model_path.count("sync") == 2:
                    ending = 9                
                if self.ssvep_mode == "async_dynamic":
                    idlepath = f"{self.ssvep_model_path[:-ending]}_idle.pth"
                elif self.ssvep_mode == "async_static":
                    idlepath = f"{self.ssvep_model_path[:-ending]}_idles.pth"                
           
                if os.path.exists(idlepath):
                    self.ssvep_class_offset = 2                
                    self.ssvep_idle_model = torch.load(idlepath, map_location=torch.device("cpu"))
                    self.ssvep_idle_model.set_device("cpu")
                    self.ssvep_idle_model.eval()            
            # generate reference by itcca method
            self.ssvep_references = self.ssvep_model        

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

    def filter_and_epoch(self, paradigm, stim):
        '''
        '''
        if paradigm == 'ERP':
            self.erp_stims_time = self.tmp_list 
            begin = self.erp_begin            
            samples = self.erp_epochDuration                        
            self.erp_y = np.array(self.erp_y)
            mrk = np.array(self.erp_stims_time).astype(int) - self.erp_begin
            step = 1
            if self.erp_stimulation == 'Multi':
                # mrk = mrk[::3]
                step = 3
            elif self.erp_stimulation == 'Dual':
                step = 2
            #if self.erp_stimulation == 'Multi':
            #        mrk = mrk[::3]
            mrk = mrk[::step] 
            low_pass = self.erp_lowPass
            high_pass = self.erp_highPass
            order = self.erp_filterOrder
            ep_dur = (np.array([0.1, 0.5])*self.fs).astype(int) # FIXME

        elif paradigm == 'SSVEP':
            begin = self.ssvep_begin                                                                                                                       
            mrk = np.array(self.ssvep_stims_time).astype(int) - begin
            samples = self.ssvep_samples
            low_pass = self.ssvep_lowPass
            high_pass = self.ssvep_highPass
            order = self.ssvep_filterOrder
            if len(self.ssvep_epochDuration) > 1:
                ep_dur = (self.ssvep_epochDuration * self.fs).astype(int)
            else:
                ep_dur = np.array([0, self.ssvep_epochDuration[0]*self.fs], dtype=int)
            # ep_dur = np.ceil(np.array([0, samples])).astype(int)
            # print("SSVEP ep dur ", ep_dur, samples)
                         
        end = int(np.floor(stim.date * self.fs))                           
        signal = eeg_filter(self.signal[:, begin:end].T, self.fs, low_pass, high_pass, order)         
        
        # ep = np.ceil(np.array([0.1,0.5])*self.fs).astype(int) # FIXME
        # erp_epochs = processing.eeg_epoch(erp_signal,ep , mrk, self.fs)
        
        # epochs = processing.eeg_epoch(signal, np.array([0, samples],dtype=int), mrk, self.fs)
        epochs = eeg_epoch(signal, ep_dur, mrk, self.fs)
        
        # del signal        
        # del mrk
        
        return epochs #.astype(np.float16)

    def erp_predict(self):
        '''
        '''
        self.erp_stims = np.array(self.erp_stims) 
        predictions = []            
        nbr = 1
        if self.erp_stimulation == 'Single':
            if self.erp_keras_model or self.erp_model_file_type == 'xml':
                if self.erp_model_file_type == 'h5':
                    predictions = self.erp_model.predict(self.erp_x.transpose((2,1,0)))
                elif self.erp_model_file_type == 'xml':
                    predictions = [self.erp_model.predict(self.erp_x.transpose((2,1,0))[i][None,...]) for i in range(self.erp_x.shape[-1])]
                    '''
                    # predictions = predict_openvino_model(self.erp_model, self.erp_x.transpose((2,1,0)))
                    for i in range(self.erp_x.shape[-1]):
                        epoch = self.erp_x.transpose((2,1,0))[i][None,...]
                        predictions.append(predict_openvino_model(self.erp_model, epoch).item())
                    '''
                else:
                    predictions = self.erp_model.predict(self.erp_x.transpose((2,1,0)), normalize=True)
                # predictions[predictions > .5] = 1.
            else:
                predictions = self.erp_model.predict(self.erp_x)
            self.command, idx = utils.select_target(predictions, self.erp_stims, commands)
        elif self.erp_stimulation == 'Dual' or self.stimulation == 'Multi':
            events = self.erp_stims
            if self.stimulation == 'Dual':
                nbr = 2
            elif self.stimulation == 'Multi':
                nbr = 3
            events = events.reshape((len(events)//nbr, nbr))
            events = np.flip(events, axis=1)
            for model in self.erp_model:
                predictions.append(model.predict(self.erp_x))
            self.command, scores = utils.select_target_multistim(np.array(predictions).T, events)
            print(scores)
            del events
        
        del predictions

    def print_if_target(self, paradigm):
        '''
        '''
        if paradigm ==  'ERP':                    
            if self.mode == 'Copy':
                self.erp_y[self.erp_y == 1] = -1
                self.erp_y[self.erp_y == 0] = 1
                tg = np.where(self.erp_y == 1)                               
                print('[ERP Target] : ', self.erp_stims[tg[0][0]] )
                self.erp_target.append(self.erp_stims[tg[0][0]]) 
                if self.command == '#':
                    print('NO ERP detection ...')
                elif int(self.command) == self.erp_stims[tg[0][0]]:                               
                    print('[ERP Correct! ]')
                    self.erp_correct += 1
        
        elif paradigm == 'SSVEP':
            self.ssvep_y = np.array(self.ssvep_y)
            # if self.mode == 'Copy':
            print('[SSVEP] preds:', self.command, ' target:', self.ssvep_y)
            if int(self.command) == self.ssvep_y:
                self.ssvep_correct += 1
            self.ssvep_target.append(self.ssvep_y[-1])

    def ssvep_predict(self):
        '''
        '''
        if self.ssvep_mode == 'sync':                
            # sync_trials = np.where(self.ssvep_y != 1)
            # ssvep_sync_epochs = ssvep_epochs[:,:,sync_trials].squeeze()
            ssvep_epochs = self.ssvep_x.squeeze()
            ssvep_sync_epochs = ssvep_epochs
            
            ssvep_predictions = self.ssvep_model.predict(ssvep_sync_epochs[0:self.ssvep_samples,:].transpose((1,0))) + self.ssvep_class_offset                             
            ssvep_predictions = np.array(ssvep_predictions)
                                
        # elif self.ssvep_mode == 'async' or self.ssvep_mode == 'sync_train':
        else:
            # ssvep_predictions = self.ssvep_model.predict(ssvep_epochs)
            if self.ssvep_keras_model:
                if self.ssvep_model_file_type == 'pth':
                    ssvep_predictions = self.ssvep_model.predict(self.ssvep_x.transpose((2, 1, 0)), normalize=True).argmax() + self.ssvep_class_offset
                else:
                    ssvep_predictions = self.ssvep_model.predict(self.ssvep_x.transpose((2, 1, 0))).argmax() + self.ssvep_class_offset
                # ssvep_predictions = self.ssvep_model.predict(self.ssvep_x[..., None].transpose((2, 1, 0))).argmax() + 1
            else:
                ssvep_predictions = self.ssvep_model.predict(self.ssvep_x)
                # ssvep_predictions = self.ssvep_model.predict(self.ssvep_x[..., None]) + 1 #TRCA

        self.command = str(ssvep_predictions.item())

    def print_results(self):
        '''
        '''
        print('Trial N :', self.n_trials)
        if self.mode == 'Copy':
            print('Trial N :', self.n_trials, ' / ERP Target : ', self.erp_target, ' / ERP Pred : ', self.erp_pred)
            # print('Trial N :', self.n_trials, ' / ERP Pred : ', self.erp_pred)
            print('Trial N :', self.n_trials, ' / SSVEP Target : ', self.ssvep_target, ' / SSVEP Pred : ', self.ssvep_pred)
            # print('Trial N :', self.n_trials, ' / SSVEP Pred : ', self.ssvep_pred)
            print('Trial N :', self.n_trials, ' / ERP Accuracy : ', (self.erp_correct / self.n_trials) * 100, '/  SSSVEP Accuracy : ', (self.ssvep_correct / self.n_trials) * 100 )                              

    def experiment_end(self):
        '''
        '''
        print('EXPERIMENT ENDS')
        self.ssvep_target = np.array(self.ssvep_target, dtype=int)
        if self.mode == 'Copy':
            print(' ERP Accuracy : ', (self.erp_correct / self.n_trials) * 100)
            print(' SSSVEP Accuracy : ', (self.ssvep_correct / self.n_trials) * 100)   
            erp_correct = np.array(self.erp_target, dtype=int) == np.array(self.erp_pred, dtype=int)
            ssvep_correct = self.ssvep_target == np.array(self.ssvep_pred, dtype=int)
            hybrid_acc = np.logical_and(erp_correct, ssvep_correct).mean()*100
            print(' Hybrid Accuracy : ', hybrid_acc)
        
        if self.ssvep_mode == "async_dynamic" or self.ssvep_mode == "async_static":
            self.ssvep_idle_cmd = np.array(self.ssvep_idle_cmd)
            y_idle = np.zeros_like(self.ssvep_target)
            y_idle[(self.ssvep_target - 1) != 0] = 1.
            idle_p = np.zeros_like(self.ssvep_idle_cmd)
            idle_p[self.ssvep_idle_cmd > .5] = 1. 
            print(f"Idle accuracy : {accuracy_score(y_idle, idle_p)*100}, AUC :{roc_auc_score(y_idle, idle_p)}")
            print(f"Idle cm: {confusion_matrix(y_idle, idle_p)}")             
                      


        self.switch = False
        del self.signal
        del self.erp_x
        del self.erp_model
        del self.ssvep_x
        del self.ssvep_model
        jitter = np.diff(self.tr_dur)
        print('Trial durations delay: ',  jitter, jitter.min(), jitter.max(), jitter.mean())
        stimSet = OVStimulationSet(0.,0.)    
        stimSet.append(OVStimulation(OpenViBE_stimulation['OVTK_StimulationId_ExperimentStop'], 0.,0.)) 
        self.output[0].append(stimSet)

    def ERP_trial(self, stim):
        '''
        '''
        if(stim.identifier == OpenViBE_stimulation['OVTK_StimulationId_TrialStart'] and self.switch):                       
            print('[ERP trial start]', stim.date)            
            self.ssvep_y = []
            self.ssvep_stims_time = []
            self.ssvep_idle_preds = [] 
            self.ssvep_idle_count = 0 
            self.ssvep_feedback = False 

            if(len(self.erp_stims_time) == 0):
                self.erp_begin = int(np.floor(stim.date * self.fs))
                        
        if (stim.identifier == OpenViBE_stimulation['OVTK_StimulationId_Target'] or 
            stim.identifier == OpenViBE_stimulation['OVTK_StimulationId_NonTarget'] and self.switch):            
            self.mode = 'Copy'
            self.erp_y.append(stim.identifier - OpenViBE_stimulation['OVTK_StimulationId_Target'])
                         
        if (stim.identifier >= OVTK_StimulationLabel_Base) and (stim.identifier <= OpenViBE_stimulation['OVTK_StimulationId_LabelEnd'] and self.switch) :
            self.erp_stims.append(stim.identifier - OVTK_StimulationLabel_Base) 
            # print('[ERP stim]', stim.date, self.erp_stims[-1])
            self.tmp_list.append(np.floor(stim.date*self.fs))        

        if(stim.identifier == OpenViBE_stimulation['OVTK_StimulationId_TrialStop'] and self.erp_y):                            
            print('[ERP trial stop]', stim.date)
            self.erp_x = self.filter_and_epoch('ERP', stim)
            self.erp_predict()
            # self.command = '1'
            # self.command = random.choice(commands)                
            print('[ERP] Command to send is: ', self.command)            
            
            self.feedback_socket.sendto(self.command.encode(), (self.hostname, self.erp_feedback_port))
            self.erp_pred.append(self.command)                       
                            
            self.print_if_target('ERP')
            self.erp_x = []
            self.tmp_list = []           
                        
            self.n_trials += 1   
            self.print_results() 
            self.switch = False
            
    def async_limit(self):
        """Test if Trial length reached the duration of 500 (or any async_dur fixed) ms (used for async mode)
        """
        if self.ssvep_stims_time:
            dur = self.signal.shape[1] - self.ssvep_stims_time[-1]  - (self.ssvep_idle_count * int(self.ssvep_async_slide*self.fs) )
            return dur >= (self.ssvep_async_dur * self.fs)      

    def ssvep_predict_idle(self, ssvep_x):
        return self.ssvep_idle_model.predict(ssvep_x.transpose((2, 1, 0)), normalize=True)
    
    def SSVEP_trial(self, stim):
        '''
        '''
        if (stim.identifier == OpenViBE_stimulation['OVTK_StimulationId_TrialStart'] and not self.switch): 
            self.erp_stims = []
            self.erp_stims_time = []
            self.erp_y = []                               
            print('[SSVEP trial start]', stim.date)
            self.tr_dur.append(stim.date)
            self.ssvep_begin = int(np.floor(stim.date * self.fs))                             
                            
        if (stim.identifier == OpenViBE_stimulation['OVTK_StimulationId_VisualSteadyStateStimulationStart'] and not self.switch):
            print('[SSVEP Visual_stim_start]', stim.date)
                            
        if (stim.identifier >= OVTK_StimulationLabel_Base) and (stim.identifier <= OVTK_StimulationLabel_Base+len(self.ssvep_frequencies) and not self.switch):
            self.ssvep_y.append(stim.identifier - OVTK_StimulationLabel_Base)
            # print('[SSVEP stim]', stim.date, self.ssvep_y[-1])
            self.ssvep_stims_time.append(np.floor(stim.date*self.fs)) 

        if(stim.identifier == OpenViBE_stimulation['OVTK_StimulationId_TrialStop'] and not self.switch):        
            print('[SSVEP trial stop]', stim.date)    

            # print('[TRIAL duration:]', stim.date - self.ssvep_stims_time[0] / 512)
            if self.ssvep_mode == "async_dynamic":
                self.ssvep_idle_cmd.append(max(self.ssvep_idle_preds)[0])

            if not self.ssvep_feedback:
                # self.ssvep_x = self.filter_and_epoch('SSVEP', stim)
                # self.ssvep_predict()
                if self.ssvep_mode == "async_static":
                    idle_p = self.ssvep_predict_idle(self.ssvep_x)                             
                    if idle_p < .5:
                        self.command = "1"                                                             
                    self.ssvep_idle_cmd.append(idle_p[0]) 
                speed = ['1', '2', '3', '4']
                self.command = random.choice(speed)
                print('[SSVEP] Sending as feedback: ', self.command)                
                self.feedback_socket.sendto(self.command.encode(), (self.hostname, self.ssvep_feedback_port))                                
                self.ssvep_pred.append(self.command)

            self.print_if_target('SSVEP')                              
            self.switch = True
                       

    def process(self):        
        
        # stream signal
        if self.input[0]:            
            self.stream_signal()

        # Dynamic SSVEP trial
        if self.ssvep_mode == 'async_dynamic':
            if self.async_limit() and self.ssvep_stims_time and not self.switch:
                self.ssvep_idle_count += 1
                # filter, epoch, predict                                
                idle_tmp_end = int(self.ssvep_stims_time[-1] + (self.ssvep_idle_count * int(self.ssvep_async_dur*self.fs)))
               
                ssvep_signal = eeg_filter(self.signal[:,self.ssvep_begin:idle_tmp_end].T, self.fs, self.ssvep_lowPass, self.ssvep_highPass, self.ssvep_filterOrder)
                mrk = np.array(self.ssvep_stims_time, dtype=int) - self.ssvep_begin + int(self.fs* self.ssvep_async_slide*(self.ssvep_idle_count-1))
                dr =  np.array([0, self.ssvep_async_dur*self.fs], dtype=int)            
                
                ssvep_x = eeg_epoch(ssvep_signal, dr, mrk, self.fs)
                self.ssvep_idle_preds.append(self.ssvep_predict_idle(ssvep_x)[0])      
                
                if len(self.ssvep_idle_preds) == 3: # 2 , 3 number of consecutive idle trials
                    idle_score = max(self.ssvep_idle_preds)
                    print(f"IDLESCORE : {idle_score}")
                    if idle_score < 0.5:
                        self.ssvep_feedback = True
                        self.feedback_socket.sendto("1".encode(), (self.hostname, self.ssvep_feedback_port))
                        print("feedback sent")                        
                        self.ssvep_pred.append(1)
                        self.command = "1"  
        
        # collect Stimulations markers and times for each paradigm
        if self.input[1]:
            chunk = self.input[1].pop()
            if type(chunk) == OVStimulationSet:
                for stimIdx in range(len(chunk)):
                    if chunk:
                        stim = chunk.pop()               
                        # SSVEP session
                        self.SSVEP_trial(stim)
                        
                        if self.switch:
                            self.ERP_trial(stim)                                                            
                                                    
                        # Ending Experiment 
                        if stim.identifier == OpenViBE_stimulation['OVTK_StimulationId_ExperimentStop']:
                            self.experiment_end()

            del chunk
        return        

    def unintialize(self):
        pass

box = HybridOnline()