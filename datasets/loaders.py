import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_credit_fraud(path: str):
    """
    Loads and preprocesses the UCI Credit Card Fraud dataset
    """
    df = pd.read_csv(path)

    X = df.drop("Class", axis=1)
    y = df["Class"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y.values
