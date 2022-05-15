from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from tensorflow.keras.models import load_model as K_load_model
from openvino.inference_engine import IECore
from pyLpov.models.openvino import OpenVinoModel
from tensorflow import keras
import tensorflow as tf
import numpy as np
import subprocess
import pickle
import torch
import h5py
import os



def is_keras_model(filepath):
    """Test whether a classifier is Keras model saved in h5 format

    Parameters
    ----------
    filepath : str
        classifier filepath
    """
    return h5py.is_hdf5(filepath)

def is_pytorch_model(filepath):
    """Test whether a classifier is a PyTorch model

    Parameters
    ----------
    filepath : str
        classifier filepath
    """
    m_type = model_type(filepath)
    return m_type == 'pth' or m_type == 'pt'   


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
    if os.path.isdir(filepath):
        file_type = 'tf'
        deep_model = True
    else:
        file_type = model_type(filepath)
        deep_model = is_keras_model(filepath) or is_pytorch_model(filepath) or file_type == 'xml'
    
    if deep_model:         
        if file_type == 'h5' or file_type == 'tf':
            # regular Keras model
            model = K_load_model(filepath)
        elif file_type == 'pth' or file_type == 'pt':
            device = available_device()
            # PyTorch Model
            if file_type == 'pt':
                file_type = 'pth'
            model = torch.load(filepath, map_location=torch.device(device))
            model.set_device(device)
            model.eval()
        elif file_type == 'xml':
            # optimized OpenVINO model
            model = load_openvino_model(filepath)
    else:  
        file_type = ''      
        model = pickle.load(open(filepath, 'rb'),  encoding='latin1') #py3
    return model, deep_model, file_type


def available_device():
    """Returns 'cuda' if an CUDA GPU is available in the env, cpu otherwise.
    """
    return 'cuda' if torch.cuda.is_available() else 'cpu'

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
    opn_model = OpenVinoModel(exec_net)
    # net.batch_size = 1
    # return exec_net
    return opn_model


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

    # we have one request only, see num_requests above
    # request = exec_net.requests[0]
    request = net.requests[0]

    # infer() waits for the result
    # for asynchronous processing check async_infer() and wait()
    request.infer({input_name: epoch})

    # read the result
    prediction_openvino_blob = request.output_blobs[output_name]
    # prediction_openvino = prediction_openvino_blob.buffer
    # return prediction_openvino
    return prediction_openvino_blob.buffer

def model_type(filepath):
    """Determine model's type (as file's extension) from filepath 

    Parameters
    ----------
    filepath : str
        model type extension
    """
    return filepath.split('/')[-1].split('.')[-1]


def freeze_model(model, frozen_folder, debug=False):
    """Convert Keras model to a TensorFlow frozen graph

    Parameters
    ----------
    model : Keras model
        Trained model
    frozen_folder : str
        path where to save frozen model
    debug : bool, optional
        if True print name of layers in the model, by default False
    """
    frozen_graph_filename = f"{model.name}_Frozen"
    pb_file_name = f"{frozen_graph_filename}.pb"
    # Convert Keras model to ConcreteFunction
    full_model = tf.function(lambda x: model(x))
    full_model = full_model.get_concrete_function(tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

    # Get frozen ConcreteFunction
    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()
   
    if debug:
        layers = [op.name for op in frozen_func.graph.get_operations()]
        print("-" * 60)
        print("Frozen model layers: ")
        for layer in layers:
            print(layer)
    
        print("-" * 60)
        print("Frozen model inputs: ")
        print(frozen_func.inputs)
        print("Frozen model outputs: ")
        print(frozen_func.outputs)

    # Save frozen graph to disk
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir=frozen_folder,
                      name=pb_file_name,
                      as_text=False)

    # Save its text representation
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir=frozen_folder,
                      name=f"{frozen_graph_filename}.pbtxt",
                      as_text=True)
    return pb_file_name

def model_optimizer(pb_file, output_dir, input_shape):
    """Generate OpenVINO optimized model by calling model optimizer script

    Parameters
    ----------
    pb_file : str
        pb frozen graph path
    output_dir : str
        folder where the optimized model will be stored
    input_shape : list of int
        model input shape eg.: [batch, channels, samples]  
    """
    # mo_tf_path = '"C:\Program Files (x86)\Intel\openvino_2021\deployment_tools\model_optimizer\mo_tf.py"'
    mo_tf_path = "C:\Program Files (x86)\Intel\openvino_2021\deployment_tools\model_optimizer\mo_tf.py"
    input_shape_str = str(input_shape).replace(' ','')
    #!python {mo_tf_path} --input_model {pb_file} --output_dir {output_dir} --input_shape {input_shape_str} --data_type FP32 --disable_nhwc_to_nchw
    cmd = subprocess.run(["python", mo_tf_path, "--input_model", pb_file, 
                         "--output_dir", output_dir, "--input_shape", input_shape_str,
                         "--data_type", "FP32", "--disable_nhwc_to_nchw"])
    print(f"The exit code was: {cmd.returncode}")

def model_optimizer_savedmodel(model_dir, output_dir, input_shape):
    """Generate OpenVINO optimized model by calling model optimizer script

    Parameters
    ----------
    model_dir : str
        tf SavedModel dir
    output_dir : str
        folder where the optimized model will be stored
    input_shape : list of int
        model input shape eg.: [batch, channels, samples]  
    """
    # mo_tf_path = '"C:\Program Files (x86)\Intel\openvino_2021\deployment_tools\model_optimizer\mo_tf.py"'
    mo_tf_path = "C:\Program Files (x86)\Intel\openvino_2021\deployment_tools\model_optimizer\mo_tf.py"
    input_shape_str = str(input_shape).replace(' ','')
    #!python {mo_tf_path} --saved_model_dir {model_dir} --output_dir {output_dir} --input_shape {input_shape_str} --data_type FP32 --disable_nhwc_to_nchw
    # --data_type "FP32"
    cmd = subprocess.run(["python", mo_tf_path, "--saved_model_dir", model_dir, 
                         "--output_dir", output_dir, "--input_shape", input_shape_str,
                         "--data_type", "FP16", "--disable_nhwc_to_nchw"])
    print(f"The exit code was: {cmd.returncode}")