from pyLpov.io.models import freeze_model, model_optimizer
from tensorflow.keras.models import load_model
import argparse
import os

parser = argparse.ArgumentParser(description="Convert trained Keras model to an OpenVINO optimized model")

# Keras Model path
parser.add_argument("Model_path",
                    metavar="model_path",
                    type=str,
                    help="Keras model full path name")
# Frozen graph folder
parser.add_argument("Frozen_folder",
                    metavar="forzen_folder",
                    type=str,
                    help="Frozen graph folder"
                    )
# Output Folder
parser.add_argument("Output_dir",
                    metavar="output_dir",
                    type=str,
                    help="optimized model folder")

args = parser.parse_args()

model_path = args.Model_path
frozen_folder = args.Frozen_folder
output_dir = args.Output_dir

# model_path = ""
# frozen_folder = ""
# pb_file = ""
# output_dir = ""

# Load Keras model
print(f"***Loading model : {model_path}")
model = load_model(model_path)
input_shape = model.input.shape.as_list()[1:]
input_shape.insert(0, 1)

# Save the model as frozen graph
# print(f"***Freezing model in: {frozen_folder}")
pb_file = freeze_model(model, frozen_folder, debug=False)
pb_file = frozen_folder + "\\" + pb_file 

# Generate OpenVINO optimized model
# print(f"***Optimizing Pb file: {pb_file} in: {output_dir}")
model_optimizer(pb_file, output_dir, input_shape)

