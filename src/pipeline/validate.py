
import pandas as pd


class DataValidationError(ValueError):
    """Raised when the dataset fails validation."""


REQUIRED_COLUMNS = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
    "target",
]


def validate_data(df: pd.DataFrame) -> bool:
    """Validate required columns, species labels, and feature ranges."""
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise DataValidationError(f"Missing required columns: {missing}")

    if df.empty:
        raise DataValidationError("Dataset is empty")

    if df[REQUIRED_COLUMNS].isnull().any().any():
        raise DataValidationError("Dataset contains missing values")

    if not df["target"].isin([0, 1, 2]).all():
        raise DataValidationError("Invalid Iris target labels")

    if not df["sepal length (cm)"].between(4.0, 8.0).all():
        raise DataValidationError("Sepal length is outside the allowed range")

    return True
