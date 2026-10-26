
import pandas as pd
from src.pipeline.preprocess import preprocess_data


def test_preprocess_removes_duplicates():
    df = pd.DataFrame({
        "feature": [1.0, 1.0, 2.0],
        "target": [0, 0, 1],
    })
    result = preprocess_data(df)
    assert len(result) == 2


def test_preprocess_fills_missing_numeric_values_with_median():
    df = pd.DataFrame({
        "feature": [1.0, None, 5.0],
        "target": [0, 1, 0],
    })
    result = preprocess_data(df)
    assert result["feature"].isna().sum() == 0
    assert result.loc[1, "feature"] == 3.0


def test_preprocess_drops_missing_target():
    df = pd.DataFrame({
        "feature": [1.0, 2.0],
        "target": [0.0, None],
    })
    result = preprocess_data(df)
    assert len(result) == 1
    assert result["target"].isna().sum() == 0


def test_preprocess_handles_empty_dataframe():
    df = pd.DataFrame(columns=["feature", "target"])
    result = preprocess_data(df)
    assert result.empty
