import numpy as np
from sklearn.metrics import confusion_matrix

def gmean(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    sensitivity = tp / (tp + fn + 1e-10)
    specificity = tn / (tn + fp + 1e-10)

    return np.sqrt(sensitivity * specificity)
