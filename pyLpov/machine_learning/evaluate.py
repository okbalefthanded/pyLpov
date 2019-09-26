from __future__ import print_function, division
# ML utils
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import confusion_matrix
# classifiers
# from swlda.swlda import SWLDA
# from blda.blda import BLDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.svm import SVC
# feature extraction
from sklearn.gaussian_process.kernels import RBF
# from blda.feature_extraction import EPFL
# from mlr.mlr import MLR
# 
from scipy.linalg import eig
from scipy import sqrt
import scipy.signal as sig
import numpy as np
# utils
import pickle
import os

'''
# Training function
def train_ERP(x, y):
    print("ERP Training...")
    # clf = make_pipeline(preprocessing.StandardScaler(), LDA(solver='lsqr', shrinkage='auto')) # Shrinkage-LDA 
    # clf = make_pipeline(preprocessing.StandardScaler(), SWLDA()) # SWLDA
    # clf = make_pipeline(preprocessing.StandardScaler(), LogisticRegression(penalty='l1', solver='liblinear') ) # LogisticRegression + L1
    # clf = make_pipeline(preprocessing.StandardScaler(), GaussianProcessClassifier(kernel=1.0 * RBF(length_scale=1.0), optimizer=None) ) # Gaussian Process
    clf = make_pipeline(EPFL(decimation_factor=12, p=0.1), preprocessing.StandardScaler(), BLDA(verbose=True)) # BLDA 
    # cv = KFold(n_splits=5)
    # scores = cross_val_score(clf, x, y, cv=cv)
    # cv_results = cross_validate(clf, x, y, cv=cv, 
                                # scoring=('accuracy', 'roc_auc'),
                                # return_train_score=True)
    clf.fit(x, y)
    print("End training")
    #return clf, cv_results
    return clf

def train_SSVEP(ssvep_x, ssvep_y):
    print("SSVEP Training...")
    ssvep_x = ssvep_x.transpose((2,1,0))
    clf = make_pipeline(MLR(),  preprocessing.StandardScaler(), SVC())
    # clf = make_pipeline(MLR(), SVC())
    cv = KFold(n_splits=5)
    cv_results = cross_validate(clf, ssvep_x, ssvep_y, cv=cv, 
                                scoring=('accuracy'),
                                return_train_score=True)
    print("End training")
    return clf, cv_results
'''
def evaluate_pipeline(x, y, pipeline):   
    # 
    # forcing x shape to be [n_samples, n_features]
    if x.shape[0] != y.shape[0]:
        x = x.transpose((2,1,0))

    if len(np.unique(y)) == 2:
        metrics = ('accuracy', 'roc_auc')
    else:
        metrics = ('accuracy')

    cv = KFold(n_splits=5)
    cv_results = cross_validate(pipeline, x, y, cv=cv, 
                                scoring=metrics,
                                return_train_score=True)

    return pipeline, cv_results