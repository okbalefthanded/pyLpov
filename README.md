# pyLpov : python-LARESI-processing-OpenVibe 

- Accompanying [LARESI_BCI](https://github.com/okbalefthanded/Laresi_BCI) Python scripts for offline/online analysis and classification of EEG data in OpenVibe 

---

### Version requirements
Python 2.7 (a limitation from OpenVibe system which only supports version 2.7 so far. OpenVibe <=2.2.0)

---

## Installation

First, clone repo from github:

```
git clone https://github.com/okbalefthanded/pyLpov.git
```
Then,  

```
cd pyLpov

pip install -r requirements.txt

python setup.py install
```
---

## Usage

### Offline processing
The API is built in a way that provides automatic processing through the use of [YAML](https://wiki.python.org/moin/YAML) configuration files and [Scikit-learn](https://scikit-learn.org/stable/) classes : [Pipeline](https://scikit-learn.org/stable/modules/compose.html#pipeline)  and [Estimators](https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html?highlight=estimator#sklearn.base.BaseEstimator) (inspired from the API of [MOABB](https://github.com/NeuroTechX/moabb))

A configuration file specifies the pipline chain of operations from preprocessing to feature extractors to classifiers.

#### Using CLI
see ```cmd_tuto.bat``` on how to set and execute the offline analysis, and follow ```sa_hybrid_train.py``` in pyLpov/scripts/standalone on how to set an automatice processing script. 

### Online processing 
- Make sure the python scripting box is available in the OpenVibe Designer Scripting tab.
- Add a python scripting box to the scenario.
- Follow this tutorial for correct usage of python scripts [Python in Openvibe](http://openvibe.inria.fr/tutorial-using-python-with-openvibe/)
- Add one of the online scripts from pyLpov/scripts/scenarios to your experiment scenario, for example use ```ssvep_py_online.py``` for SSVEP online detection.

---
## Methods available

As pyLpov API relies heavily on scikit-learn, any built-in classifier or regressor can be easily defined in the pipeline, the same goes for any 3rd-party methods developed with scikit-learn's estimators. Nevertheles we keep adding specific BCI methods, the following list shows the available methods so far:

### Event-Related Potentials (ERP)
- Downsample and vector concatenation.
- [EPFL](http://infoscience.epfl.ch/record/101093) approach

### Steady-State Visual Evoked Potentials (SSVEP) 
- CCA and [ITCCA](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0140703) 
- [MLR](https://ieeexplore.ieee.org/abstract/document/7389413/)
---

## Citation

---

## Acknowledgment

