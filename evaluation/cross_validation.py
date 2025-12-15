from sklearn.model_selection import StratifiedKFold
import numpy as np

def stratified_cv(model, X, y, metric_fn, use_proba=False):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []

    for train_idx, test_idx in skf.split(X, y):
        model.fit(X[train_idx], y[train_idx])

        if use_proba:
            y_scores = model.predict_proba(X[test_idx])[:, 1]
            score = metric_fn(y[test_idx], y_scores)
        else:
            y_pred = model.predict(X[test_idx])
            score = metric_fn(y[test_idx], y_pred)

        scores.append(score)

    return np.array(scores)
