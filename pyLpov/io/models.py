from tensorflow.keras.models import load_model as K_load_model
from openvino.inference_engine import IECore
import h5py
import pickle

def is_keras_model(filepath):
    """Test whether a classifier is Keras model saved in h5 format

    Parameters
    ----------
    filepath : str
        classifier filepath
    """
    return h5py.is_hdf5(filepath)   


def load_model(filepath):
    """load classifier saved in filepath, a model can be either a sklearn pipeline pickle
    object or a H5 Keras model.

    Parameters
    ----------
    filepath : str
        model's path
    
    Returns
    -------
        - sklearn pipeline object OR
        - Keras H5 saved model object OR
        - Openvino Inference Engine network object
    """
    file_type = model_type(filepath)
    deep_model = is_keras_model(filepath)
    if deep_model or file_type== 'xml':        
        if file_type == 'h5':
            # regular Keras model
            model = K_load_model(filepath)
        elif file_type == 'xml':
            # optimized OpenVINO model
            model = load_openvino_model(filepath)
    else:  
        file_type = ''      
        model = pickle.load(open(filepath, 'rb'),  encoding='latin1') #py3
    return model, deep_model, file_type


def load_openvino_model(filepath):
    """load openvino optimized model as inference engine network

    Parameters
    ----------
    filepath : str
        openvino model xml file path
    
    Returns
    -------
    inference engine network
    """
    model_xml = filepath    
    fname = filepath.split('/')[-1].split('.')[0]
    path = filepath.split('/')
    path = path[:-1]
    path.append(fname)
    path = '/'.join(path)
    model_bin = '.'.join([path, 'bin'])
    ie = IECore()
    net = ie.read_network(model=model_xml, weights=model_bin)
    exec_net = ie.load_network(network=net, device_name="CPU")
    # net.batch_size = 1
    return exec_net


def predict_openvino_model(net, epoch):
    """Make prediction on epoch using openvino net

    Parameters
    ----------
    net : Openvino executable network object
        
    epochs : numpy ndarray [n_trials x channels x samples], n_trials is equal to batch size 
    in net.
        n_trials = 1 for SSVEP experiment
        n_trials = Number of items in an ERP experiment        

    Returns
    -------
    ndarray [1, number of classes]
        models probabilites output
    """
    input_name = next(iter(net.input_info))
    output_name = next(iter(net.outputs))
    # ie = IECore()
    # number of request can be specified by parameter num_requests, default 1
    # exec_net = ie.load_network(network=net, device_name="CPU")

    # we have one request only, see num_requests above
    # request = exec_net.requests[0]
    request = net.requests[0]

    # infer() waits for the result
    # for asynchronous processing check async_infer() and wait()
    request.infer({input_name: epoch})

    # read the result
    prediction_openvino_blob = request.output_blobs[output_name]
    prediction_openvino = prediction_openvino_blob.buffer
    return prediction_openvino

def model_type(filepath):
    """Determine model's type (as file's extension) from filepath 

    Parameters
    ----------
    filepath : str
        model type extension
    """
    return filepath.split('/')[-1].split('.')[-1]