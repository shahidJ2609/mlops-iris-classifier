
import pandas as pd


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean an Iris-like dataset by removing duplicates and handling missing values."""
    if df.empty:
        return df.copy()

    result = df.copy()
    result = result.drop_duplicates()

    # Remove rows where the target is missing, if a target column exists.
    if "target" in result.columns:
        result = result.dropna(subset=["target"])
    elif "species" in result.columns:
        result = result.dropna(subset=["species"])

    # Fill missing numeric feature values with their column median.
    target_col = "target" if "target" in result.columns else "species"
    numeric_cols = result.select_dtypes(include="number").columns
    feature_cols = [col for col in numeric_cols if col != target_col]

    for col in feature_cols:
        if result[col].isna().any():
            result[col] = result[col].fillna(result[col].median())

    return result
