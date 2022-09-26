from torchmetrics.functional.classification.calibration_error import calibration_error
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from aawedha.io.experimental.laresi_pilot_erp import LaresiEEG
from sklearn.metrics import roc_auc_score, matthews_corrcoef
from sklearn.metrics import precision_score, recall_score

# from pyLpov.io.laresi import LaresiEEG
from pyLpov.utils import utils
import torch

cmds = [1, 2, 3, 4, 5, 6, 7, 8, 9] 


def load_dataset(load_path):
    ds = LaresiEEG()
    ds.generate_set(load_path, epoch=[.1, .5], band=[5., 12.])
    return ds

def character_selection(pred, events):
    pr = []
    for tr in range(0, len(events), 9):
        for i in range(tr, tr+9):
            command, idx = utils.select_target(pred[tr:tr+9], events[tr:tr+9], cmds)
        if command == '#':
            command = 0
        pr.append(command)
    return pr

def train_pipeline(epochs, y, clf):
    """Fit a Sklearn pipeline on epochs and y

    Parameters
    ----------
    epochs : Numpy Tensor (samples x channels x trials)
        epoched EEG trials
    y : 1d array
        epochs labels
    clf : Sklearn pipeline
        processing steps from preprocessing to classification.
    """
    clf.fit(epochs, y)

def evaluate(epochs, y, events, clf, verbose=0):
    """Evaluate test data

    Parameters
    ----------
    epochs : Numpy Tensor (samples x channels x trials) 
        epoched EEG trials
    y : 1d array
        epochs labels
    clf : trained Sklearn pipeline
        processing steps from preprocessing to classification.
    verbose : int, optional
        print results if True (=1), by default 0
    Returns
    -------
    dict
        Metric values
    """

    pred = clf.predict(epochs)
    score = clf.predict_proba(epochs)
    acc = balanced_accuracy_score(y, pred) * 100
    if score.ndim > 1 and score.shape[1] == 2:
        score = score[:, 1]
    auc_score = roc_auc_score(y, score) 
    pres = precision_score(y, pred) * 100
    phrase = events[y == 1]
    pr = character_selection(score, events)
    pr_acc = accuracy_score(phrase, pr)*100
    rec = recall_score(y, pred) * 100
    mcc = matthews_corrcoef(y, pred)
    ece = calibration_error(torch.tensor(score), torch.tensor(y, dtype=int)).numpy().item()
    metrics = { 'char': pr_acc,
                'ba':   acc,
                'auc': auc_score,
                'recall': rec,
                #'brier': brier_score_loss(y, yp, pos_label=1),
                'mcc': mcc,
                'ece':  ece
              }
    if verbose:
        print(f"Accuracy: {acc} // AUC: {auc_score} // Precision: {pres} // Recall: {rec} // Correct Character Selection : {pr_acc}")
    return metrics


def train_pipeline_single(ds, clf):
    clf.fit(ds.epochs.squeeze(), ds.y.squeeze())


def evaluate_single(ds_test, clf, verbose=0):
    # clf.fit(ds.epochs.squeeze(), ds.y.squeeze())
    pred = clf.predict(ds_test.epochs.squeeze())
    # pred = clf.predict_proba(ds_test.epochs.squeeze()).argmax(axis=1)
    score = clf.predict_proba(ds_test.epochs.squeeze()) #.argmax(axis=1)
    # acc = accuracy_score(ds_test.y.squeeze(), pred) * 100
    acc = balanced_accuracy_score(ds_test.y.squeeze(), pred) * 100
    # auc_score = roc_auc_score(ds_test.y.squeeze(), pred)
    if score.ndim > 1 and score.shape[1] == 2:
        score = score[:, 1]
    auc_score = roc_auc_score(ds_test.y.squeeze(), score) 
    pres = precision_score(ds_test.y.squeeze(), pred) * 100
    phrase = ds_test.events[0, ds_test.y[0] == 1]
    # pr = character_selection(pred, ds_test.events)
    pr = character_selection(score, ds_test.events)
    rec = recall_score(ds_test.y.squeeze(), pred) * 100
    pr_acc = accuracy_score(phrase, pr)*100    
    if verbose:
        print(f"Accuracy: {acc} // AUC: {auc_score} // Precision: {pres} // Recall: {rec} // Correct Character Selection : {pr_acc}")
    # print("Confusion matrix:", confusion_matrix(ds_test.y.squeeze(), pred))
    return acc, auc_score, pres, rec, pr_acc
