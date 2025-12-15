# Systematic Benchmarking of Machine Learning Models on Imbalanced Datasets

## 📌 Overview

Many real-world machine learning problems such as **fraud detection**, **medical diagnosis**, and **customer churn prediction** involve **severely imbalanced datasets**. In these scenarios, traditional accuracy-based evaluation is misleading and can result in incorrect conclusions.

This repository provides a **research-grade, reproducible benchmarking framework** to evaluate machine learning models on imbalanced datasets using **robust metrics** and **statistical significance testing**.

---

## 🎯 Objectives

* Benchmark machine learning models fairly on imbalanced datasets
* Implement metrics suitable for skewed class distributions
* Perform statistically valid model comparisons
* Provide a clean, extensible evaluation pipeline

---

## 🧠 Key Contributions

* Custom implementation of **G-Mean**, **Matthews Correlation Coefficient (MCC)**, and **PR-AUC**
* **Stratified K-Fold Cross-Validation** for reliable evaluation
* **Wilcoxon Signed-Rank** and **Friedman tests** for statistical significance
* Modular and reproducible experimental design

---

## 📂 Repository Structure

```
imbalanced-ml-benchmark/
│
├── data/
│   └── raw/
│
├── datasets/
│   └── loaders.py
│
├── models/
│   ├── logistic.py
│   ├── random_forest.py
│   └── xgboost.py
│
├── metrics/
│   ├── gmean.py
│   ├── mcc.py
│   └── pr_auc.py
│
├── evaluation/
│   ├── cross_validation.py
│   └── statistical_tests.py
│
├── experiments/
│   └── run_experiment.py
│
├── requirements.txt
└── README.md
```

---

## 📊 Evaluation Metrics

### 1️⃣ G-Mean

Balances sensitivity and specificity and is defined as:

[ G\text{-}Mean = \sqrt{TPR \times TNR} ]

### 2️⃣ Matthews Correlation Coefficient (MCC)

A correlation-based metric robust to class imbalance:

[ MCC \in [-1, 1] ]

### 3️⃣ Precision–Recall AUC (PR-AUC)

Preferred over ROC-AUC for highly imbalanced datasets, as it focuses on the minority class performance.

---

## 🔄 Cross-Validation Strategy

* **Stratified K-Fold (k = 5)**
* Preserves class ratios in each fold
* Identical splits used across all models for fair comparison

---

## 📈 Statistical Significance Testing

To ensure that observed performance differences are **not due to randomness**, the following tests are used:

* **Wilcoxon Signed-Rank Test** for pairwise model comparison
* **Friedman Test** for comparing multiple models

A p-value < 0.05 indicates statistically significant differences.

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/imbalanced-ml-benchmark.git
cd imbalanced-ml-benchmark
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Download Dataset

Download the **UCI Credit Card Fraud Detection Dataset** and place it in:

```
data/raw/creditcard.csv
```

### 4️⃣ Run Experiments

```bash
python experiments/run_experiment.py
```

---

## 🧪 Models Implemented

* Logistic Regression (class-weighted)
* Random Forest (class-weighted)
* XGBoost (cost-sensitive learning)

---

## 📌 Example Output

```
Logistic      | GMean=0.61 | MCC=0.39 | PR-AUC=0.36
RandomForest  | GMean=0.72 | MCC=0.55 | PR-AUC=0.52
XGBoost       | GMean=0.78 | MCC=0.63 | PR-AUC=0.61

Wilcoxon Test (RF vs XGB on PR-AUC)
p-value = 0.018
```

---

## 🔬 Future Extensions

* SMOTE vs cost-sensitive learning comparison
* Nemenyi post-hoc analysis
* Multi-class imbalance benchmarking
* Time-series imbalanced datasets
* LaTeX-ready result tables for publication

---

## 👤 Intended Audience

* Data Scientists
* Machine Learning Engineers
* ML Researchers
* MSc / PhD students

---

## 📜 License

MIT License

---

## ⭐ Citation

If you use this repository for academic or professional work, please cite the project.

