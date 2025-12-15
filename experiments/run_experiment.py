from datasets.loaders import load_credit_fraud
from models.logistic import get_model as lr_model
from models.random_forest import get_model as rf_model
from models.xgboost import get_model as xgb_model

from metrics.gmean import gmean
from metrics.mcc import mcc
from metrics.pr_auc import pr_auc

from evaluation.cross_validation import stratified_cv
from evaluation.statistical_tests import wilcoxon_test

import numpy as np

X, y = load_credit_fraud("data/raw/creditcard.csv")

models = {
    "Logistic": lr_model(),
    "RandomForest": rf_model(),
    "XGBoost": xgb_model()
}

results = {}

for name, model in models.items():
    print(f"Running {name}...")
    g = stratified_cv(model, X, y, gmean)
    m = stratified_cv(model, X, y, mcc)
    p = stratified_cv(model, X, y, pr_auc, use_proba=True)

    results[name] = {
        "GMean": g,
        "MCC": m,
        "PRAUC": p
    }

    print(f"{name} | GMean={g.mean():.3f} | MCC={m.mean():.3f} | PR-AUC={p.mean():.3f}")

# Statistical comparison
print("\nWilcoxon Test (RF vs XGB) on PR-AUC")
stat, p = wilcoxon_test(results["RandomForest"]["PRAUC"],
                        results["XGBoost"]["PRAUC"])
print(f"Statistic={stat:.4f}, p-value={p:.4f}")
