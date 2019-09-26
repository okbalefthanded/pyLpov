"""
    SSVEP paradigm 

"""
from pyLpov.paradigms.base import Paradigm

frequency_stimulation = {0: 'ON_OFF', 1:'SIN'} # stimulation mode

class SSVEP(Paradigm):
    """
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

    phrase : str
        sequence of characters to be spelled by the subject during the experiments

    stim_type : str
        type of stimulations used in the experiment : ON_OFF / Sampled_Sinusoidal /

    frequencies : list
        frequencies presented at each stimuli in Hz
    
    phase : list
        phase of each frequency in rad

    Methods
    -------
    """
      
    def __init__(self, title='SSVEP_LARESI', paradigmType='SSVEP', experimentMode='CALIBRATION', controlMode=0, 
                stimulationDuration=4000, breakDuration=4000, nrSequences=10, desiredPhrase='',
                nrElements=4, stimulationMode=0, frequencies=['7.5','8.57','10','12'], phase=None
                ):
        super(SSVEP, self).__init__(title, paradigmType, experimentMode, controlMode, 
                                    stimulationDuration, breakDuration, nrSequences,
                                    desiredPhrase, nrElements 
                                    ) 
        self.stimulationMode = frequency_stimulation[stimulationMode]                                   
        self.frequencies = frequencies
        self.phase = phase

