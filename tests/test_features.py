
import pandas as pd
import pytest

from src.pipeline.features import engineer_features


@pytest.fixture
def sample_data():
    return pd.DataFrame({
        "sepal length (cm)": [5.0, 6.0],
        "sepal width (cm)": [3.0, 2.5],
        "petal length (cm)": [2.0, 4.0],
        "petal width (cm)": [0.5, 1.5],
        "target": [0, 1],
    })


def test_sepal_area(sample_data):
    result = engineer_features(sample_data)
    assert result["sepal_area"].tolist() == [15.0, 15.0]


def test_petal_area(sample_data):
    result = engineer_features(sample_data)
    assert result["petal_area"].tolist() == [1.0, 6.0]


def test_sepal_to_petal_length_ratio(sample_data):
    result = engineer_features(sample_data)
    assert result["sepal_to_petal_length_ratio"].tolist() == [2.5, 1.5]


def test_petal_length_bin(sample_data):
    result = engineer_features(sample_data)
    assert result["petal_length_bin"].astype(str).tolist() == [
        "short", "medium"
    ]


def test_zero_petal_length_produces_nan():
    df = pd.DataFrame({
        "sepal length (cm)": [5.0],
        "petal length (cm)": [0.0],
    })
    result = engineer_features(df)
    assert pd.isna(result.loc[0, "sepal_to_petal_length_ratio"])
