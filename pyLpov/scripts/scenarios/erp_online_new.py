from pyLpov.proc.processing import eeg_filter, eeg_epoch
from pyLpov.scripts.online import OnLine
from pyLpov.io.models import load_model
from pyLpov.utils import utils
import numpy as np

commands = ['1', '2', '3', '4', '5', '6', '7', '8', '9']

class ERPOnline(OVBox, OnLine):

    def __init__(self):
        OVBox.__init__(self)
        OnLine.__init__(self, paradigm="ERP")
        self.downsample = 0
        self.movingAverage = 1
        self.stimulation = "Single"

    # OnLine specific Methods
    def filter_and_epoch(self, stim):
        '''
        '''
        self.stims_time = self.tmp_list 
        self.stims = np.array(self.stims)                          
        self.end = int(np.floor(stim.date * self.fs)) 
        self.y = np.array(self.y) 
        mrk = np.array(self.stims_time).astype(int) - self.begin
        step = 1
        if self.stimulation == 'Multi':
            step = 3
        elif self.stimulation == 'Dual':
            step = 2
        mrk = mrk[::step]
        erp_signal = eeg_filter(self.signal[:, self.begin:self.end].T, self.fs, self.lowPass, self.highPass, self.filterOrder)        
        strt = int(0.1*self.fs)
        # np.array([strt, self.erp_epochDuration],dtype=int)
        # ep = np.ceil(np.array([0.1, 0.5])*self.fs).astype(int) # FIXME
        ep = (np.array([0.1, 0.5])*self.fs).astype(int) # FIXME
        self.x = eeg_epoch(erp_signal, ep, mrk, self.fs) #.astype(np.float16)
        del erp_signal
        del mrk

    def single_stimulation(self, commands):
        '''
        '''
        if self.deep_model:
            predictions = self.predict_deep_model()
        else:
            predictions = self.model.predict(self.x)
        self.command, idx = utils.select_target(predictions, self.stims, commands)
        del predictions

    def multiple_stimulation(self, commands):
        '''
        '''
        predictions = []
        nbr = 1    
        events = np.array(self.stims)
        if self.stimulation == 'Dual':
            nbr = 2
        elif self.stimulation == 'Multi':
            nbr = 3
        events = events.reshape((len(events)//nbr, nbr))
        events = np.flip(events, axis=1)
        for model in self.model:
            predictions.append(model.predict(self.x))
        self.command, scores = utils.select_target_multistim(np.array(predictions).T, events)
        if self.command == '0':
            self.command = '#' # there is no 0 command it's a padding with command 5
        print(scores)
        del events
        del predictions

    def predict(self, commands):
        '''
        '''    
        if self.stimulation == 'Single':
            self.single_stimulation(commands)
        
        # TODO: multiple stimulation 
        elif self.stimulation == 'Dual' or self.stimulation == 'Multi':
            self.multiple_stimulation(commands)

        self.pred.append(int(self.command))        

    def print_if_target(self):
        '''
        '''               
        if self.experiment_mode == 'Copy':
            self.y[self.y == 1] = -1
            self.y[self.y == 0] = 1
            target_index = np.where(self.y == 1)[0][0]                               
            print('[ERP Target] : ', self.stims[target_index] )
            self.target.append(self.stims[target_index]) 
            if self.command == '#':
                print('NO ERP detection ...')
            elif int(self.command) == self.stims[target_index]:                               
                print('[ERP Correct! ]')
                self.correct += 1
    
    def experiment_end(self):
        '''
        '''
        self.terminate()
        stimSet = OVStimulationSet(0.,0.)    
        stimSet.append(OVStimulation(OpenViBE_stimulation['OVTK_StimulationId_ExperimentStop'], 0.,0.)) 
        self.output[0].append(stimSet)   
    
    # OVBox specific Methods
    def initialize(self):
        self.fs = int(self.setting["Sample Rate"])
        #
        self.lowPass = int(self.setting["ERP Low Pass"])
        self.highPass = int(self.setting["ERP High Pass"])
        self.filterOrder = int(self.setting["ERP Filter Order"])
        self.downSample = int(self.setting["Downsample Factor"])
        self.epoch_duration = np.ceil(float(self.setting["ERP Epoch Duration (in sec)"]) * self.fs).astype(int)
        self.movingAverage = int(self.setting["ERP Moving Average"])
        self.model_path = self.setting["Classifier"]
        self.model, self.deep_model, self.model_file_type = load_model(self.model_path)
        self.stimulation = str(self.setting["Stimulation"])

    def process(self):        
        
        # stream signal
        if self.input[0]:            
            self.stream_signal(self.input[0])        
        
        # collect Stimulations markers and times for each paradigm
        if self.input[1]:
            chunk = self.input[1].pop()
            if type(chunk) == OVStimulationSet:
                for stimIdx in range(len(chunk)):
                    if chunk:
                        stim = chunk.pop()               
                        # ERP session                        
                        if(stim.identifier == OpenViBE_stimulation['OVTK_StimulationId_TrialStart']):                       
                            print('[ERP trial start]', stim.date)
                            self.tr_dur.append(stim.date)
                            self.set_begin(stim)                              
                        
                        if (stim.identifier == OpenViBE_stimulation['OVTK_StimulationId_Target'] or 
                            stim.identifier == OpenViBE_stimulation['OVTK_StimulationId_NonTarget']): 
                            self.experiment_mode = 'Copy'
                            self.y.append(stim.identifier - OpenViBE_stimulation['OVTK_StimulationId_Target'])
                         
                        if (stim.identifier >= utils.OVTK_StimulationLabel_Base) and (stim.identifier <= OpenViBE_stimulation['OVTK_StimulationId_LabelEnd']) :
                            self.stims.append(stim.identifier - utils.OVTK_StimulationLabel_Base) 
                            self.tmp_list.append(np.floor(stim.date*self.fs))        

                        if(stim.identifier == OpenViBE_stimulation['OVTK_StimulationId_TrialStop']):                         
                            print('[ERP trial stop]', stim.date)  

                            self.decode(stim, commands)
                            self.feedback()                                             
                            self.post_trial()                       

                        # Ending Experiment 
                        if stim.identifier == OpenViBE_stimulation['OVTK_StimulationId_ExperimentStop']:
                            self.experiment_end()

            del chunk
        return       

    def unintialize(self):
        pass


box = ERPOnline()