from __future__ import print_function, division
import numpy as np
import pickle
import socket


OVTK_StimulationLabel_Base = 0x00008100
commands = ['1', '2', '3', '4', '5', '6', '7', '8', '9']


# calculate score for each stimulus and select the target
def select_target(predictions, events):

    scores = []
    array = np.array(events)
    values = set(array)

    for i in range(1, len(values) + 1):
        item_index = np.where(array == i)
        cl_item_output = np.array(predictions)[item_index]          
        score = np.sum(cl_item_output == 0) / len(cl_item_output)
        scores.append(score)
        
        # check if the target returned by classifier is out of the commands set
        # or all stimuli were classified as non-target
    if scores.count(0) == len(scores):
    #    print self.scores
        feedback_data = '#'
    else:
       feedback_data = commands[scores.index(max(scores))]

    return feedback_data


class Predictor(OVBox):

    def __init__(self):
        super(Predictor, self).__init__()
        self.model = None
        self.predictions = []
        self.events = []
        self.trials_count = 0
        self.feedback_data = 0
        self.hostname = '127.0.0.1'
        self.feedback_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.feedback_port = 12345

    def initialize(self):
        # Load model
        model_filename = self.setting['classifier']
        self.model = pickle.load(open(model_filename, 'rb'))


    def process(self):
        
        # recieve feature vectors
        # classify each epoch       
        if self.input[0]:
            x = self.input[0].pop()
            if type(x) == OVStreamedMatrixBuffer:
                if (x):
                    feature_vector = np.array(x).reshape(1, -1)
                    self.predictions.append(self.model.predict(feature_vector))

        if self.input[1]:
            chunk = self.input[1].pop()
            if type(chunk) == OVStimulationSet:
                for stimIdx in range(len(chunk)):
                    if chunk:
                        stim = chunk.pop()
                        
                        # 
                        if stim.identifier ==  OpenViBE_stimulation['OVTK_StimulationId_TrialStart']:
                            self.events = []
                            self.predictions = []
                            self.trials_count += 1
                        
                        # print('Received Marker: ', stim.identifier, 'stamped at', stim.date, 's')
                        if (stim.identifier >= OVTK_StimulationLabel_Base) and (stim.identifier <= OpenViBE_stimulation['OVTK_StimulationId_LabelEnd']):
                            self.events.append(stim.identifier - OVTK_StimulationLabel_Base)
                            
                        if stim.identifier == OpenViBE_stimulation['OVTK_StimulationId_TrialStop']:
                            # make final decision
                            command = select_target(self.predictions, self.events)                            
                            self.feedback_socket.sendto(command, (self.hostname, self.feedback_port))
                            print('The command is:', command)                          
                     
                    
        
    def uninitialize(self):
        pass



box = Predictor()