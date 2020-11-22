from pyLpov.utils.StimulationsCodes import OpenViBE_stimulation
from pyLpov.proc.processing import eeg_filter, eeg_epoch
from pyLpov.paradigms.base import Paradigm
import numpy as np
import pandas as pd

Base_Stimulations = 0x00008100

class DataSet(object):

    def __init__(self, epochs=None, channels= None, fs = None, y=None, 
                ev_desc=None, ev_pos=None, paradigm=None, session_interval = []
                ):
        self.epochs = epochs
        self.channels = channels
        self.fs = fs
        self.y = y
        self.ev_desc = ev_desc
        self.ev_pos = ev_pos
        self.paradigm = paradigm
        self.session_interval = session_interval

    def get_epochs(self, cnt, flt_opts):
        #
        if isinstance(self.session_interval[0], float):
            self.session_interval = self.session_interval.astype(int)
        # epoch_duration, high_pass, low_pass, filter_order = flt_opts.values()
        epoch_duration, low_pass, high_pass, filter_order = flt_opts.values()
        epoch_duration = int(np.floor(epoch_duration * self.fs))
        # filter
        signal = cnt[self.session_interval[0]:self.session_interval[1],:]
        signal = eeg_filter(signal, self.fs, low_pass, high_pass, filter_order) 
        cnt[self.session_interval[0]:self.session_interval[1],:] = signal
        # epoch
        epochs = []
        if self.paradigm.paradigmType == 'ERP':
            self.ev_pos = self.ev_pos.reshape((self.paradigm.stimuli, self.paradigm.stimuli*self.paradigm.nrSequences))
            for i in range(self.ev_pos.shape[0]):
                epochs.append(eeg_epoch(cnt, np.array([0, epoch_duration]), self.ev_pos[i,:]))
            epochs = np.array(epochs).transpose((1,2,3,0))
        elif self.paradigm.paradigmType == 'SSVEP':
            epochs = eeg_epoch(cnt, np.array([0, epoch_duration]), self.ev_pos)
        #
        self.epochs = epochs

    @staticmethod
    def convert_raw(datapath, prdg):
        raw = pd.read_csv(datapath)
        fs = int(raw.columns[0].split(':')[1].split('Hz')[0])
        chs = [ch for ch in raw.columns if len(ch) <= 3]
        
        cnt = raw[chs].to_numpy()
        raw_desc = DataSet.get_events(raw, 'Event Id')
        raw_pos = DataSet.get_events(raw, 'Event Date') 

        start_interv = raw_pos[raw_desc==OpenViBE_stimulation['OVTK_StimulationId_ExperimentStart']]
        end_interv =  raw_pos[raw_desc==OpenViBE_stimulation['OVTK_StimulationId_ExperimentStop']]

        if end_interv.size > 1:
            # Hybrid paradigm
            start2 = start_interv[start_interv> end_interv[0]][0]
            starts = np.nditer(np.array([start_interv[0], start2]))
            ends = np.nditer(end_interv)
            ds = []           
            for k in prdg.__dict__.keys():
                inst = prdg.__getattribute__(k)
                if isinstance(inst, Paradigm):
                    ds.append(DataSet.construct_dataset(inst, raw_desc, raw_pos, fs, chs, starts.next(), ends.next()) )
        else:         
            ds = DataSet.construct_dataset(prdg, raw_desc, raw_pos, fs, chs, start_interv[0], end_interv)

        return cnt, ds #, session_intrval

    @staticmethod
    def construct_dataset(prdg, raw_desc, raw_pos, fs, chs, starts, ends):
        stimuli = prdg.stimuli
        
        idx = np.logical_and(raw_pos >=starts, raw_pos <=ends)
        desc = raw_desc[idx]
        pos = raw_pos[idx]
        
        id_stim = np.logical_and(desc > Base_Stimulations,  desc <= Base_Stimulations + stimuli)
        desc = desc[id_stim] - Base_Stimulations
        pos = np.floor(pos[id_stim] * fs).astype(int)
        
        if prdg.paradigmType == 'ERP':
            y = raw_desc[np.logical_or(raw_desc==OpenViBE_stimulation['OVTK_StimulationId_Target'], raw_desc==OpenViBE_stimulation['OVTK_StimulationId_NonTarget'])]
            y[y==OpenViBE_stimulation['OVTK_StimulationId_Target']] = 1
            y[y==OpenViBE_stimulation['OVTK_StimulationId_NonTarget']] = -1 
        elif prdg.paradigmType == 'SSVEP':
            y = desc     
               
        session_interval = np.floor( [starts, ends] ) * fs # begin, end of session
        
        # FIXME
        ds = DataSet(channels=chs, fs=fs, y=y, ev_desc=desc, ev_pos=pos, paradigm=prdg, session_interval=session_interval)
        return ds

    @staticmethod
    def get_events(dataframe, key):
        events_id = dataframe[key].notna()
        events = dataframe[key].loc[events_id]
        events = events.to_numpy()
        ev = [elm.split(':') for elm in events]
        ev = np.array(list(pd.core.common.flatten(ev)), dtype=float)
        return ev   
        

