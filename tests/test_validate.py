
import pandas as pd
import pytest

from src.pipeline.validate import DataValidationError, validate_data


@pytest.fixture
def valid_data():
    return pd.DataFrame({
        "sepal length (cm)": [5.1, 6.0],
        "sepal width (cm)": [3.5, 2.8],
        "petal length (cm)": [1.4, 4.5],
        "petal width (cm)": [0.2, 1.5],
        "target": [0, 1],
    })


def test_clean_data_passes(valid_data):
    assert validate_data(valid_data) is True


def test_missing_column_raises_error(valid_data):
    df = valid_data.drop(columns=["petal width (cm)"])
    with pytest.raises(DataValidationError):
        validate_data(df)


def test_invalid_species_label_raises_error(valid_data):
    valid_data.loc[0, "target"] = 5
    with pytest.raises(DataValidationError):
        validate_data(valid_data)


def test_out_of_range_sepal_length_raises_error(valid_data):
    valid_data.loc[0, "sepal length (cm)"] = 10.0
    with pytest.raises(DataValidationError):
        validate_data(valid_data)
