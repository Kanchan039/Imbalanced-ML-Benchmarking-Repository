from sklearn.linear_model import LogisticRegression

def get_model():
    return LogisticRegression(
        class_weight="balanced",
        solver="liblinear",
        max_iter=1000,
        random_state=42
    )
