"""
    ERP paradigm 

"""

from pyLpov.paradigms.base import Paradigm

FLASHING_MODE = {0: 'SC', 1:'RC'}
SPELLER_TYPE  = {0:'FLASHING_SPELLER', 1:'FACES_SPELLER', 2:'INVERTED_FACE',
                 3:'COLORED_FACE', 4:'INVERTED_COLORED_FACE', 5:'SSVEP',
                 6: 'ARABIC_SPELLER', 7: 'MULTI_STIM', 8 : 'DUAL_STIM'} # stimulation type


class ERP(Paradigm):
    """
    Attributes
    ----------
    title : str
        paradigm title : ERP / Motor Imagery / SSVEP

    control : str
        stimulation mode of paradigm : synchrounous  / asynchrounous

    stimulation : int
        duration of a single stimulation in msec

    isi : int
        inter stimulation interval : duration of stimulation pause between two consecutive stimuli in msec

    repetition : int
        number of stimulations per trial

    stimuli : int
        number of stimulus presented in the experiment

    phrase : str
        sequence of characters to be spelled by the subject during the experiments

    stim_type : str
        stimulus presented to subject: flash / face / inverted_face ...

    flashing_mode : str
        whether stimuli are presented in a row-column (RC) fashion or single character (SC)

    Methods
    -------
    """

    def __init__(self, title='LARESI_BCI', paradigmType='ERP', experimentMode='CALIBRATION', controlMode=0, 
                stimulationDuration=100, breakDuration=100, nrSequences=10, desiredPhrase='12345', stimuli=9, 
                flashingMode='SC', stimulationType='FLASHING_SPELLER'
                ):
        
        super(ERP, self).__init__(title, paradigmType, experimentMode, controlMode, stimulationDuration, 
                                  breakDuration, nrSequences, desiredPhrase, stimuli
                                 )         
        self.flashingMode = FLASHING_MODE[flashingMode]
        self.stimulationType = SPELLER_TYPE[stimulationType] 
         

    