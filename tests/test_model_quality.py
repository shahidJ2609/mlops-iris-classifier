import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


FEATURE_COLS = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]

MIN_ACCEPTABLE_ACCURACY = 0.90


def test_iris_dataset_has_expected_columns():
    df = pd.read_csv("data/processed/iris_features.csv")

    expected_columns = FEATURE_COLS + ["target"]

    assert list(df.columns) == expected_columns


def test_iris_dataset_has_150_rows():
    df = pd.read_csv("data/processed/iris_features.csv")

    assert len(df) == 150


def test_model_quality_on_iris_features():
    df = pd.read_csv("data/processed/iris_features.csv")

    X = df[FEATURE_COLS]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        random_state=42,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    assert accuracy >= MIN_ACCEPTABLE_ACCURACY, (
        f"Model accuracy {accuracy:.4f} is below "
        f"the minimum threshold of {MIN_ACCEPTABLE_ACCURACY}"
    )


def test_model_probabilities_are_valid():
    df = pd.read_csv("data/processed/iris_features.csv")

    X = df[FEATURE_COLS]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    )

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)

    assert np.all(probabilities >= 0)
    assert np.all(probabilities <= 1)
    assert np.allclose(
        probabilities.sum(axis=1),
        1.0,
        atol=1e-6,
    )
