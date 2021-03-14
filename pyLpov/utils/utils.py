from glob import glob
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
import os
import yaml
import numpy as np

# calculate score for each stimulus and select the target
def select_target(predictions, events, commands):
    scores = []
    # print('events : ', events)
    array = np.array(events)
    values = set(array)

    for i in range(1, len(values) + 1):
        item_index = np.where(array == i)
        cl_item_output = np.array(predictions)[item_index]
           
        # score = np.sum(cl_item_output == 1) / len(cl_item_output)
        score = np.sum(cl_item_output) / len(cl_item_output)
        # score = np.sum(cl_item_output)
        scores.append(score)
        
    # check if the target returned by classifier is out of the commands set
    # or all stimuli were classified as non-target
    # print(' Predictions: ', predictions)
    # print(' Scores: ', scores)
    if scores.count(0) == len(scores):
    #    print self.scores
        feedback_data = '#'
    else:
        feedback_data = commands[scores.index(max(scores))]

    return feedback_data, scores.index(max(scores))


def select_target_multistim(predictions, events):
    '''
    '''
    scores = predictions.sum(axis=0) / np.unique(events).size
    stim_type = scores.argmax()
    evs = events[predictions[:,stim_type] == 1, stim_type]
    unique, counts = np.unique(evs, return_counts=True)
    if counts.size == 0:
        feedback_data = '#'
    else:
        feedback_data = unique[counts.argmax()].astype(int)
    return str(feedback_data), scores


def parse_config_file(config_file):
    # yaml_files = ['config.yml']
    pipelines = {}
     
    with open(config_file, 'r') as _file:
        content = _file.read()
        # load config
        config_dict = yaml.load(content)
    
    config_elements = config_dict.keys()
    # check if preprocessing is defined in the config file
    if ('ERP_preprocess' in config_elements) and ('SSVEP_preprocess' in config_elements):
        pipelines['ERP_filter'] = config_dict['ERP_preprocess'].pop().get('parameters')
        pipelines['SSVEP_filter'] = config_dict['SSVEP_preprocess'].pop().get('parameters')
    else:
        pipelines['filter'] = config_dict['preprocess'].pop().get('parameters')
    
    # 
    if ('ERP_pipeline' in config_elements) and ('SSVEP_pipeline' in config_elements):
        # Hybrid approach
        ERP_pipeline = create_pipeline(config_dict['ERP_pipeline'])
        SSVEP_pipeline = create_pipeline(config_dict['SSVEP_pipeline'])
        pipelines['ERP_pipeline'] = ERP_pipeline
        pipelines['SSVEP_pipeline'] = SSVEP_pipeline        
    else:
        pipelines['pipeline'] = create_pipeline(config_dict['pipeline'])
    return pipelines    
    
def create_pipeline(config):
    # create component
    if type(config) == list: # ordinary case, pipeline elements in a list
        components = create_components_from_config(config)
    
    elif type(config) == dict: # GridSearch case, elements in a dict: (GridSearchCV, Pipeline)
        gridsearch = create_components_from_config(config['gridsearch'])
        pipe = create_components_from_config(config['pipeline'])
        clf = GridSearchCV(estimator=gridsearch[1], param_grid=gridsearch[2], cv=gridsearch[0], verbose=True)
        components = pipe + [clf]
    
    # make pipeline
    pipeline = make_pipeline_from_components(components)
    return pipeline    

def create_components_from_config(config):
    components = []

    for component in config:        
        if component['name'] == 'params':
            instance = component['parameters']
        else:
            # load the package
            mod = __import__(component['from'], fromlist=[component['name']])        
            # create the instance
            if 'parameters' in component.keys():
                params = component['parameters']
            else:
                params = {}
            instance = getattr(mod, component['name'])(**params)        
        components.append(instance)
    
    return components

def make_pipeline_from_components(components):
    pipeline = make_pipeline(*components)
    return pipeline
