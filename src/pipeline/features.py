
import numpy as np
import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features to an Iris dataset."""
    result = df.copy()

    if "sepal length (cm)" in result and "sepal width (cm)" in result:
        result["sepal_area"] = (
            result["sepal length (cm)"] * result["sepal width (cm)"]
        )

    if "petal length (cm)" in result and "petal width (cm)" in result:
        result["petal_area"] = (
            result["petal length (cm)"] * result["petal width (cm)"]
        )

    if "sepal length (cm)" in result and "petal length (cm)" in result:
        result["sepal_to_petal_length_ratio"] = (
            result["sepal length (cm)"]
            / result["petal length (cm)"].replace(0, np.nan)
        )

    if "petal length (cm)" in result:
        result["petal_length_bin"] = pd.cut(
            result["petal length (cm)"],
            bins=[-np.inf, 2, 5, np.inf],
            labels=["short", "medium", "long"],
        )

    return result
