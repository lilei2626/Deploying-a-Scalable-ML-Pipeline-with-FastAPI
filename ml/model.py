from __future__ import annotations
import pickle
from sklearn.metrics import fbeta_score, precision_score, recall_score
from ml.data import process_data
# TODO: add necessary import


from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

# Optional: implement hyperparameter tuning.
# --------- Training ---------
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.
    """
    # Bump max_iter to reduce convergence warnings; keep class_weight for imbalance
    model = LogisticRegression(max_iter=2000, class_weight="balanced")
    model.fit(X_train, y_train)
    return model


# --------- Metrics / Inference ---------
def compute_model_metrics(y, preds) -> Tuple[float, float, float]:
    """
    Compute precision, recall, and F1.
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """
    Run model inferences and return predictions.
    """
    return model.predict(X)


# --------- Persistence ---------
def save_model(model, path):
    """
    Serialize model (or encoder/lb) to a file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path):
    """
    Load a previously saved object.
    """
    path = Path(path)
    return joblib.load(path)


# --------- Slice performance ---------
def performance_on_categorical_slice(
    model: Any,
    data,                      # pandas.DataFrame
    categorical_feature: str,  # e.g., "education"
    value: Any,                # e.g., "Bachelors"
    categorical_features: list[str],
    label: str,
    encoder: Any,
    lb: Any,
) -> Tuple[float, float, float]:
    """
    Compute precision, recall, and F1 on the subset where
    `categorical_feature == value`.
    Returns (precision, recall, f1).
    """
    # 1) build the slice dataframe
    df_slice = data[data[categorical_feature] == value]
    if df_slice.empty:
        return 0.0, 0.0, 0.0

    # 2) transform using the SAME encoders fit on training
    X_slice, y_slice, _, _ = process_data(
        df_slice,
        categorical_features=categorical_features,
        label=label,
        training=False,
        encoder=encoder,
        lb=lb,
    )

    # 3) predict and score
    preds = inference(model, X_slice)
    p, r, fb = compute_model_metrics(y_slice, preds)
    return float(p), float(r), float(fb)