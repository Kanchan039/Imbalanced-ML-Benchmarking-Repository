from sklearn.metrics import precision_recall_curve, auc

def pr_auc(y_true, y_scores):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    return auc(recall, precision)
