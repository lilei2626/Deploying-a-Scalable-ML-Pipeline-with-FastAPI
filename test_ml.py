import pytest
# TODO: add necessary import
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics

# TODO: implement the first test. Change the function name and input as needed

DATA_PATH = "data/census.csv"

def test_process_data_shapes():
    """
    Test that process_data returns arrays of the correct shape and type.
    """
    df = pd.read_csv(DATA_PATH)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["salary"])

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    X_train, y_train, encoder, lb = process_data(
        train_df, categorical_features=cat_features, label="salary", training=True
    )
    X_test, y_test, _, _ = process_data(
        test_df, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
    )

    assert X_train.shape[0] == len(train_df)
    assert X_test.shape[0] == len(test_df)
    assert y_train.shape[0] == len(train_df)
    assert y_test.shape[0] == len(test_df)
    assert isinstance(X_train, np.ndarray)
    assert isinstance(y_train, np.ndarray)


def test_train_model_returns_logistic_regression():
    """
    Test that train_model returns a scikit-learn LogisticRegression model.
    """
    X = np.array([[0, 1], [1, 0], [1, 1], [0, 0]])
    y = np.array([0, 1, 1, 0])
    model = train_model(X, y)
    assert isinstance(model, LogisticRegression)


def test_compute_model_metrics_known_case():
    """
    Test that compute_model_metrics returns correct values for a simple case.
    y = [0,1,1,0], preds = [0,1,0,0]
      TP=1, FP=0, FN=1 => precision=1.0, recall=0.5, f1=0.666...
    """
    y = np.array([0, 1, 1, 0])
    preds = np.array([0, 1, 0, 0])
    precision, recall, f1 = compute_model_metrics(y, preds)

    assert pytest.approx(precision, rel=1e-6) == 1.0
    assert pytest.approx(recall, rel=1e-6) == 0.5
    assert pytest.approx(f1, rel=1e-6) == 2 * (1.0 * 0.5) / (1.0 + 0.5)