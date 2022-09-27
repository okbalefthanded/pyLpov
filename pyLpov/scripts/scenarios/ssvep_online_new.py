
from pyLpov.utils.experiment import serialize_experiment, update_experiment_info
from pyLpov.proc.processing import eeg_filter, eeg_epoch
from pyLpov.scripts.online import OnLine
from pyLpov.io.models import load_model
from baseline.ssvep.cca import CCA
from pyLpov.utils import utils
from copy import deepcopy
import numpy as np

# np.set_printoptions(precision=4)
OVTK_StimulationLabel_Base = 0x00008100

class SSVEPpredictor(OVBox, OnLine):

    def __init__(self):
        # super(SSVEPpredictor, self).__init__()
        OVBox.__init__(self)
        OnLine.__init__(self, paradigm="SSVEP")
        # self.frequencies = ['idle', 16, 16.75, 17.25, 18]
        # self.frequencies = ['idle', 9, 9.75, 10.5, 11.25]
        # self.frequencies = ['idle', 9.25, 9.75, 10.25, 10.75]
        self.frequencies = ['idle', 8, 9, 10, 11]
        # self.frequencies = ['idle', 8, 9, 10]
        # self.frequencies = ['idle', 10, 9, 8]
        # self.frequencies = ['idle', 11, 10, 9, 8]
        # self.frequencies = ['idle', 9.5, 9, 8.5, 8]
        # self.frequencies = ['idle', 8.57, 6.67, 12, 5.45]
        # self.frequencies = ['idle', 8, 8.75, 9.5, 10.25]
        # self.frequencies = ['idle', 8.25, 9, 9.75, 10.5]
        #
        self.async_trial = False
        self.mode = 'sync'      
        self.harmonics = 0
        self.references = []
        #       
        self.file_name = ''
        self.is_replay = False

    # OnLine specific Methods
    def filter_and_epoch(self, stim):
        '''
        '''
        self.stims = np.array(self.stims)                          
        self.y = np.array(self.y) 
        mrk = np.array(self.stims_time) - self.begin        
        ssvep_signal = eeg_filter(self.signal[:, self.begin:self.end].T, self.fs, self.low_pass, self.high_pass, self.filter_order)       
        
        '''
        if (mrk + self.dur.max()) > ssvep_signal.shape[0]:
            dur = np.array([0. ,0.5 * self.fs], dtype=int)
        else:
            dur = self.dur
        '''
        dur = self.dur
        self.x = eeg_epoch(ssvep_signal, dur, mrk, self.fs).squeeze() #.astype(np.float16)
        # self.ssvep_x  = eeg_epoch(ssvep_signal, self.dur, mrk, self.fs, baseline_correction=True).squeeze()
        del ssvep_signal
        del mrk        
    
    def predict(self):
        '''
        '''
        if self.mode == 'sync':                
            # sync_trials = np.where(self.ssvep_y != 1)
            # ssvep_sync_epochs = ssvep_epochs[:,:,sync_trials].squeeze()
            # ssvep_sync_epochs = self.ssvep_x
            ssvep_predictions = np.array(self.model.predict(self.x[0:self.samples, :].transpose((1,0))) + 1)
                             
        elif self.mode == 'async' or self.mode == 'sync_train':
            if self.deep_model:   
                ssvep_predictions = self.predict_deep_model().argmax() + 1
            else:
                ssvep_predictions = self.model.predict(self.x[..., None]) + 1
            
            # ssvep_predictions = np.array([1])
        self.command = str(ssvep_predictions.item())
        self.pred.append(int(self.command))

    def print_if_target(self):
        '''
        '''
        if self.experiment_mode == 'Copy':
            print('[SSVEP] preds:', self.command, ' target:', self.y)
            if int(self.command) == self.y:
                self.correct += 1
            self.target.append(self.y[-1])

    def experiment_end(self):
        '''
        '''
        print('EXPERIMENT ENDS')
        # update experiment json file
        if not self.is_replay:
            update_experiment_info(self.file_name, "repetition", self.n_trials // len(self.frequencies))
        
        self.terminate()

        stimSet = OVStimulationSet(0.,0.)    
        stimSet.append(OVStimulation(OpenViBE_stimulation['OVTK_StimulationId_ExperimentStop'], 0., 0.)) 
        self.output[0].append(stimSet)     
       
    def async_limit(self):
        """Test if Trial length reached the duration of 500 ms (used for async mode)
        """
        # dur = self.signal.shape[1] - self.ssvep_begin
        if self.stims_time:
            dur = self.signal.shape[1] - self.stims_time[-1]
            self.async_trial = True
            return dur >= (0.5 * self.fs)        
    
    # OVBox specific Methods
    def initialize(self):
        #
        if 'File Name' in self.setting:
            self.file_name = f"{self.setting['File Name']}.json"
        else:
            self.is_replay = True
        self.experiment_mode = self.setting["Experiment Mode"]
        self.fs = int(self.setting['Sample Rate'])
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
            # generate reference signals
            x = [ [np.cos(2*np.pi*f*t*i),np.sin(2*np.pi*f*t*i)] for f in frequencies for i in range(1, self.harmonics+1)]
            self.references = np.array(x).reshape(len(frequencies), 2*self.harmonics, self.samples)
            self.model = CCA(self.harmonics, frequencies, phase=None, references=self.references, length=int(dur))
            
        elif self.mode == 'async' or self.mode == 'sync_train':
            self.model, self.deep_model, self.model_file_type = load_model(self.model_path)
            # generate reference by itcca method
            self.references = self.model
        # create json file
        info = {'title': "SSVEP_LARESI",
                'stimulation' : int(self.epoch_duration[1]*1000),
                'break_duration' : int(self.epoch_duration[0]*1000),
                'repetition': 0,
                'stimuli' : len(self.frequencies),
                'phrase' : [],
                'stim_type' : 'Sinusoidal',
                'frequencies' : self.frequencies,
                'control' : self.mode 
                }
        if not self.is_replay:
            serialize_experiment(self.file_name, info)
    
    def process(self):
        
        # stream signal
        if self.input[0]:            
            self.stream_signal(self.input[0])        
        '''
        if self.mode == 'async':
            if self.async_limit():
                # print("Reached 500 ms", self.signal.shape)
                # predict
                # FIXME
                self.feedback_socket.sendto("-1".encode(), (self.hostname, self.feedback_port)) 
        '''
        
        # collect Stimulations markers and times for each paradigm
        if self.input[1]:
            chunk = self.input[1].pop()
            if type(chunk) == OVStimulationSet:
                for stimIdx in range(len(chunk)):
                    if chunk:
                        stim = chunk.pop()               
                        # SSVEP session                        
                        if(stim.identifier == OpenViBE_stimulation['OVTK_StimulationId_TrialStart']):                       
                            print('[SSVEP trial start]', stim.date)
                            self.tr_dur.append(stim.date)
                             # self.ssvep_begin = int(np.floor(stim.date * self.fs))
                            if(len(self.stims_time) == 0):                             
                                self.begin = int(stim.date * self.fs)

                        if (stim.identifier == OpenViBE_stimulation['OVTK_StimulationId_VisualSteadyStateStimulationStart']):
                                print('[SSVEP Visual_stim_start]', stim.date)
                                self.stims_time.append(int(stim.date*self.fs))

                        if (stim.identifier >= utils.OVTK_StimulationLabel_Base) and (stim.identifier <= utils.OVTK_StimulationLabel_Base+len(self.frequencies)):
                                self.y.append(stim.identifier - OVTK_StimulationLabel_Base) 
                                print('[SSVEP stim]', stim.date, self.y[-1])                               
                        
                        if(stim.identifier == OpenViBE_stimulation['OVTK_StimulationId_TrialStop']):                           
                            print('[SSVEP trial stop]', stim.date)
                            self.stim_dur.append(stim.date - self.stims_time[0] / self.fs)
                            print('[Stim duration:]', self.stim_dur[-1])
                            self.end = int(stim.date * self.fs)                             
                            
                            self.decode(stim)                            
                            self.feedback()
                            self.post_trial()                     

                        # Ending Experiment 
                        if stim.identifier == OpenViBE_stimulation['OVTK_StimulationId_ExperimentStop']:
                            self.experiment_end()

            del chunk
        return        

    def unintialize(self):
        pass

box = SSVEPpredictor()
