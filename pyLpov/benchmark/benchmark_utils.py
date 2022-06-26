from aawedha.io.experimental.laresi_pilot_erp import LaresiEEG
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import roc_auc_score, confusion_matrix
# from pyLpov.io.laresi import LaresiEEG
from pyLpov.utils import utils

cmds = [1, 2, 3, 4, 5, 6, 7, 8, 9] 

def load_dataset(load_path):
    ds = LaresiEEG()
    ds.generate_set(load_path, epoch=[.1, .5], band=[5., 12.])
    return ds

def character_selection(pred, events):
    pr = []
    for tr in range(0, len(events[0]), 9):
        for i in range(tr, tr+9):
            command, idx = utils.select_target(pred, events[0, tr:tr+9], cmds)
        if command == '#':
            command = 0
        pr.append(command)
    return pr

def train_pipeline(ds, clf):
    clf.fit(ds.epochs.squeeze(), ds.y.squeeze())


def evaluate(ds_test, clf, verbose=0):
    # clf.fit(ds.epochs.squeeze(), ds.y.squeeze())
    pred = clf.predict(ds_test.epochs.squeeze())
    # pred = clf.predict_proba(ds_test.epochs.squeeze()).argmax(axis=1)
    score = clf.predict_proba(ds_test.epochs.squeeze()) #.argmax(axis=1)
    # acc = accuracy_score(ds_test.y.squeeze(), pred) * 100
    acc = balanced_accuracy_score(ds_test.y.squeeze(), pred) * 100
    auc_score = roc_auc_score(ds_test.y.squeeze(), pred) * 100
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
