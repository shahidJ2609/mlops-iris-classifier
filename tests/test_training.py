import numpy as np
from sklearn.metrics import accuracy_score

from src.train import load_data, train_model, evaluate_model


def test_load_data_has_expected_shape():
    X_train, X_test, y_train, y_test = load_data()

    assert X_train.shape[1] == 4
    assert X_test.shape[1] == 4
    assert len(X_train) + len(X_test) == 150
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)


def test_train_model_returns_model():
    X_train, X_test, y_train, y_test = load_data()

    model = train_model(X_train, y_train)

    assert hasattr(model, "fit")
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")


def test_model_prediction_shape():
    X_train, X_test, y_train, y_test = load_data()

    model = train_model(X_train, y_train)

    predictions = model.predict(X_test)

    assert predictions.shape == y_test.shape
    assert len(predictions) == len(y_test)


def test_model_predictions_are_valid_labels():
    X_train, X_test, y_train, y_test = load_data()

    model = train_model(X_train, y_train)

    predictions = model.predict(X_test)

    assert set(predictions).issubset(set(np.unique(y_train)))


def test_prediction_probabilities_sum_to_one():
    X_train, X_test, y_train, y_test = load_data()

    model = train_model(X_train, y_train)

    probabilities = model.predict_proba(X_test)

    assert probabilities.shape == (len(X_test), 3)
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)


def test_evaluate_model_returns_accuracy_and_report():
    X_train, X_test, y_train, y_test = load_data()

    model = train_model(X_train, y_train)

    accuracy, report = evaluate_model(model, X_test, y_test)

    assert 0.0 <= accuracy <= 1.0
    assert isinstance(report, str)
    assert "precision" in report


def test_model_meets_minimum_accuracy():
    X_train, X_test, y_train, y_test = load_data()

    model = train_model(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    assert accuracy >= 0.90
