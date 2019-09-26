"""
    Base class for paradigms
"""
from abc import ABCMeta, abstractmethod

PARADIGM_TYPE  = {0: 'ERP', 1: 'SSVEP', 2:'HYBRID'}   
CONTROL_MODE   = {0: 'SYNC', 1:'ASYNC'}    
OPERATION_MODE = {0: 'CALIBRATION', 1:'COPY_MODE', 2:'FREE_MODE', 3:'SSVEP_SINGLE'}   # experiment mode


class Paradigm(object):
    """
    Paradigms:

    Attributes
    ----------
    title : str
        paradigm title : ERP / Motor Imagery / SSVEP

    control : str
        stimulation mode of paradigm : synchrounous  / asynchrounous

    stimulation : int
        duration of a single stimulation in msec

    break_duration : int
        duration of stimulation pause between two consecutive stimuli in msec

    repetition : int
        number of stimulations per trial (ERP) / session (SSVEP)

    stimuli : int
        number of stimulus presented in the experiment.

    stim_type: str
        stimulus presented to subject (ERP) / type of stimulations used in the experiment (SSVEP) 

    phrase : str
        sequence of characters to be spelled by the subject during the experiments

    Methods
    -------
    """
    # python 2.7 
    # __metaclass__ = ABCMeta   


    def __init__(self, title=None, paradigmType=None, experimentMode=None, controlMode=None, stimulationDuration=0, 
                breakDuration=0, nrSequences=0, desiredPhrase=None, stimuli=0
                ):
        self.title = title
        self.paradigmType = PARADIGM_TYPE[paradigmType]
        self.experimentMode = OPERATION_MODE[experimentMode]
        self.controlMode = CONTROL_MODE[controlMode]

        self.stimulationDuration = stimulationDuration
        self.breakDuration = breakDuration
        self.nrSequences = nrSequences
        self.desiredPhrase = desiredPhrase
        self.stimuli = stimuli  

    @classmethod
    def from_json(cls, json_data):
        return cls(**json_data)