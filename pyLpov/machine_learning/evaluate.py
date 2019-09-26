from __future__ import print_function, division
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import confusion_matrix
import numpy as np


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