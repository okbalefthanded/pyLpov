from __future__ import print_function
from pyLpov.utils import utils
from pyLpov.analysis.calibration import Calibration
from pyLpov.analysis.cbhybrid import CBHYBRID
import argparse
import os

parser = argparse.ArgumentParser(description='Train and save pipelines for LARESI Hybrid/Single ERP/SSVEP BCI')

parser.add_argument('Data',
                    metavar='data',
                    type=str,
                    help='calibration data filepath')
# experiment config
parser.add_argument('Exp',
                    metavar='experiment',
                    type=str,
                    help='experiment configuration filepath'
                    )
# analysis config
parser.add_argument('Config',
                    metavar='config',
                    type=str,
                    help='approach pipelines configration filepath')

parser.add_argument('Outdir',
                    metavar='outdir',
                    type=str,
                    help='Directory where fitted pipelines are stored')

args = parser.parse_args()

data_path = args.Data
experiment = args.Exp
config_path = args.Config
config_path = os.path.abspath('../../config/' + config_path)
outdir = args.Outdir

# parse conf file
pipelines = utils.parse_config_file(config_path)
# redirect training
if pipelines.__len__() > 2:
    # hybrid analysis
    analysis = CBHYBRID(data_path, experiment, pipelines, outdir)
else:
    # single paradigm analysis
    analysis = Calibration(data_path, experiment, pipelines, outdir)
analysis.run_analysis()
analysis.save()

